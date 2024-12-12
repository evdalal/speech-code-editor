"""
# Project Name: Speech to Code
# Author: STC Team
# Date: 12/12/2024
# Last Modified: 12/12/2024
# Version: 1.0

# Copyright (c) 2024 Brown University
# All rights reserved.

# This file is part of the STC project.
# Usage of this file is restricted to the terms specified in the
# accompanying LICENSE file.

"""

# Standard Library Imports
import os
import time
import queue
import threading
import logging
import signal as system_signal  # Handling of asynchronous events (e.g., signals)
import collections
import copy
import traceback
import gc  # Garbage collector interface
import re
from enum import Enum

# Type Hinting Imports
from typing import Iterable, List, Optional, Union

# Third-Party Library Imports
import numpy as np
import torch
import torch.multiprocessing as mp  # Multiprocessing capabilities within PyTorch
import faster_whisper  # Library for faster whisper model implementations

# Signal Processing Imports
from scipy.signal import resample  # Signal resampling function
from scipy import signal  # Signal processing utilities

import config
from utils import (initialize_logging, set_up_webrtc, set_up_silero, set_multiprocessing_start_method,
                   start_thread, find_suffix_match_in_text)

# Enable handling of duplicate OpenMP libraries (intended for development and debugging only)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class TranscriptionWorker:
    def __init__(self, conn, stdout_pipe,
                 model_path, compute_type, gpu_device_index, device,
                 ready_event, shutdown_event, interrupt_stop_event,
                 task, beam_size, initial_prompt, suppress_tokens):
        self.conn = conn  # Connection object for communication with the parent process
        self.stdout_pipe = stdout_pipe

        self.model_path = model_path
        self.compute_type = compute_type
        self.gpu_device_index = gpu_device_index
        self.device = device

        self.ready_event = ready_event
        self.shutdown_event = shutdown_event
        self.interrupt_stop_event = interrupt_stop_event

        self.task = task
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens
        self.queue = queue.Queue()  # Queue to hold incoming data for transcription

    def custom_print(self, *args, **kwargs):
        """
        Custom print method to redirect print statements to a pipe.
        """
        message = ' '.join(map(str, args))
        try:
            self.stdout_pipe.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass

    def poll_connection(self):
        """
        Poll the connection for new data and place it in the processing queue.
        Continuously checks if there's data to be received from the parent process.
        """
        while not self.shutdown_event.is_set():
            if self.conn.poll(0.01):
                try:
                    data = self.conn.recv()
                    self.queue.put(data)
                except Exception as e:
                    logging.error(f"Error receiving data from connection: {e}")
            else:
                time.sleep(config.PROCESS_SLEEP_INTERVAL)

    def run(self):
        """
        Main loop to initialize the model, process incoming data, and handle transcription tasks.
        """
        if __name__ == "__main__":
            system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

        __builtins__['print'] = self.custom_print

        print(f"Initializing Faster Whisper Main transcription model from path: "
              f"'{self.model_path}' on device: '{self.device}' using compute type: '{self.compute_type}'")

        try:
            model = faster_whisper.WhisperModel(
                model_size_or_path=self.model_path,
                device=self.device,
                compute_type=self.compute_type,
                device_index=self.gpu_device_index,
            )
        except Exception as e:
            logging.exception(f"Error initializing Faster Whisper Main transcription model: {e}")
            raise

        self.ready_event.set()

        print("Faster Whisper Main transcription model initialized successfully")

        # Start a background thread to poll for incoming data
        polling_thread = threading.Thread(target=self.poll_connection)
        polling_thread.start()

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Wait for an audio item from the queue
                    audio, language = self.queue.get(timeout=0.1)
                    try:
                        segments, info = model.transcribe(
                            audio,
                            task=self.task,
                            language=language if language else None,
                            beam_size=self.beam_size,
                            initial_prompt=self.initial_prompt,
                            suppress_tokens=self.suppress_tokens
                        )
                        transcription = " ".join(seg.text for seg in segments).strip()
                        print(f"Final text detected with main model: {transcription}")
                        self.conn.send(('success', (transcription, info)))
                    except Exception as e:
                        logging.error(f"General error in transcription: {e}")
                        self.conn.send(('error', str(e)))
                except queue.Empty:
                    continue  # No data in queue, continue the loop
                except KeyboardInterrupt:
                    self.interrupt_stop_event.set()
                    logging.debug("Transcription worker process finished due to KeyboardInterrupt")
                    break
                except Exception as e:
                    logging.error(f"General error in processing queue item: {e}")
        finally:
            __builtins__['print'] = print  # Restore the original print function
            self.conn.close()
            self.stdout_pipe.close()
            self.shutdown_event.set()  # Ensure the polling thread will stop
            polling_thread.join()  # Wait for the polling thread to finish


class AudioTranscriber:
    """
    A class responsible for detecting
    voice activity, and then transcribing the captured audio using the
    Faster Whisper model.
    """

    class State(Enum):
        INACTIVE = "inactive"
        LISTENING = "listening"
        RECORDING = "recording"
        TRANSCRIBING = "transcribing"

    def __init__(self,
                 model: str = config.MODEL_TRANSCRIPTION_TYPE,
                 task: str = config.MODEL_TASK_TYPE,
                 language: str = "",
                 compute_type: str = "default",
                 input_device_index: int = None,
                 gpu_device_index: Union[int, List[int]] = 0,
                 on_recording_start=None,
                 on_recording_stop=None,

                 add_period_to_sentence=True,

                 logging_level=logging.INFO,

                 # Realtime transcription parameters
                 enable_realtime_transcription=False,
                 realtime_model_type=config.MODEL_REALTIME_TRANSCRIPTION_TYPE,
                 realtime_processing_pause=config.REALTIME_PROCESSING_PAUSE,
                 on_realtime_transcription_update=None,
                 on_realtime_transcription_stabilized=None,

                 # Voice activation parameters
                 silero_sensitivity: float = config.SILERO_VAD_SENSITIVITY,
                 silero_use_onnx: bool = True,
                 silero_inactivity_detection: bool = False,
                 webrtc_sensitivity: int = config.WEBRTC_VAD_SENSITIVITY,
                 post_speech_silence_time: float = config.POST_SPEECH_SILENCE_TIME,
                 min_recording_duration: float = config.MIN_RECORDING_DURATION,
                 min_recording_gap: float = config.MIN_RECORDING_GAP,
                 pre_recording_buffer_time: float = config.PRE_RECORDING_BUFFER_TIME,
                 on_vad_detect_start=None,
                 on_vad_detect_stop=None,

                 on_recorded_chunk=None,
                 handle_buffer_overflow: bool = config.HANDLE_BUFFER_OVERFLOW,
                 beam_size: int = 5,
                 beam_size_realtime: int = 3,
                 buffer_size: int = config.AUDIO_BUFFER_SIZE,
                 sample_rate: int = config.AUDIO_SAMPLE_RATE,
                 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
                 suppress_tokens=None,
                 early_transcription_on_silence: int = 0,
                 max_allowed_latency: int = config.MAX_ALLOWED_LATENCY,
                 disable_log_file: bool = True
                 ):

        self.task = task
        self.language = language
        self.compute_type = compute_type
        self.input_device_index = input_device_index
        self.gpu_device_index = gpu_device_index

        self.add_period_to_sentence = add_period_to_sentence

        self.min_recording_gap = min_recording_gap
        self.min_recording_duration = min_recording_duration
        self.pre_recording_buffer_time = pre_recording_buffer_time
        self.post_speech_silence_time = post_speech_silence_time
        self.on_recording_start = on_recording_start
        self.on_recording_stop = on_recording_stop
        self.on_vad_detect_start = on_vad_detect_start
        self.on_vad_detect_stop = on_vad_detect_stop
        self.on_recorded_chunk = on_recorded_chunk
        self.enable_realtime_transcription = enable_realtime_transcription
        self.main_model_type = model
        self.realtime_model_type = realtime_model_type
        self.realtime_processing_pause = realtime_processing_pause
        self.on_realtime_transcription_update = (
            on_realtime_transcription_update
        )
        self.on_realtime_transcription_stabilized = (
            on_realtime_transcription_stabilized
        )
        self.handle_buffer_overflow = handle_buffer_overflow
        self.beam_size = beam_size
        self.beam_size_realtime = beam_size_realtime
        self.max_allowed_latency = max_allowed_latency

        self.logging_level = logging_level
        self.audio_queue = mp.Queue()
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.recording_start_time = 0
        self.recording_end_timestamp = 0

        self.silero_check_time = 0
        self.silero_working = False
        self.silence_start_time_after_speech = 0
        self.silero_sensitivity = silero_sensitivity
        self.silero_inactivity_detection = silero_inactivity_detection
        self.listen_start = 0

        self.state = self.State.INACTIVE
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.webrtc_voice_detected = False
        self.silero_voice_detected = False
        self.recording_thread = None
        self.realtime_thread = None
        self.audio_interface = None
        self.audio = None
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.last_transcription_bytes = None
        self.initial_prompt = initial_prompt
        self.suppress_tokens = suppress_tokens or [-1]

        self.detected_language = None
        self.detected_language_probability = 0
        self.detected_realtime_language = None
        self.detected_realtime_language_probability = 0
        self.transcription_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.transcribe_count = 0
        self.silence_to_trigger_transcription = early_transcription_on_silence

        # Initialize the logging configuration with the specified level
        initialize_logging(console_logging_level=logging_level, disable_log_file=disable_log_file)

        self.is_shut_down = False
        self.shutdown_event = mp.Event()

        set_multiprocessing_start_method()

        logging.info("Starting Real-Time Speech to Text")

        self.interrupt_stop_event = mp.Event()
        self.was_interrupted = mp.Event()
        self.refined_transcription_ready_event = mp.Event()
        self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
        self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()

        # Determine the device for running the model (use GPU if available, otherwise fallback to CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.transcript_process = start_thread(
            target=AudioTranscriber._transcription_worker,
            args=(
                child_transcription_pipe,
                child_stdout_pipe,
                model,
                self.compute_type,
                self.gpu_device_index,
                self.device,
                self.refined_transcription_ready_event,
                self.shutdown_event,
                self.interrupt_stop_event,
                self.task,
                self.beam_size,
                self.initial_prompt,
                self.suppress_tokens
            )
        )

        # Initialize the real-time transcription model
        if self.enable_realtime_transcription:
            try:
                self.realtime_model_type = faster_whisper.WhisperModel(
                    model_size_or_path=self.realtime_model_type,
                    device=self.device,
                    compute_type=self.compute_type,
                    device_index=self.gpu_device_index
                )
            except Exception as e:
                logging.exception(f"Error initializing Real-Time Faster Whisper realtime transcription model: {e}")
                raise

            logging.info("Real-Time Faster Whisper transcription model initialized successfully")

        # Setup voice activity detection model WebRTC
        try:
            self.webrtc_vad_model = set_up_webrtc(webrtc_sensitivity)
        except Exception as e:
            logging.exception(f"Error initializing WebRTC voice activity detection engine: {e}")
            raise
        logging.info("WebRTC VAD voice activity detection engine initialized successfully")

        # Setup voice activity detection model Silero VAD
        try:
            self.silero_vad_model = set_up_silero(silero_use_onnx)
        except Exception as e:
            logging.exception(f"Error initializing Silero VAD voice activity detection engine: {e}")
            raise
        logging.info("Silero VAD voice activity detection engine initialized successfully")

        self.audio_buffer = collections.deque(
            maxlen=int((self.sample_rate // self.buffer_size) *
                       self.pre_recording_buffer_time)
        )
        self.frames = []

        # Recording control flags
        self.recording_active = False
        self.transcriber_active = True

        self.auto_start_on_voice = False
        self.auto_stop_on_silence = False

        # Start the recording worker thread
        self.recording_thread = threading.Thread(target=self._recording_worker)
        self.recording_thread.daemon = True
        self.recording_thread.start()

        # Start the real-time transcription worker thread
        self.realtime_thread = threading.Thread(target=self._realtime_worker)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()

        # Wait for transcription models to start
        logging.info("Waiting for the refined transcription model to initialize")
        self.refined_transcription_ready_event.wait()
        logging.info("Refined transcription model is ready")

        self.stdout_thread = threading.Thread(target=self._monitor_stdout)
        self.stdout_thread.daemon = True
        self.stdout_thread.start()

        logging.debug('Real-Time Speech to Text initialization completed successfully')

    def _monitor_stdout(self):
        while not self.shutdown_event.is_set():
            try:
                # Check if there is any message in the stdout pipe
                if self.parent_stdout_pipe.poll(0.1):
                    message = self.parent_stdout_pipe.recv()
                    logging.info(f"Receive from stdout pipe {message}")
            except (BrokenPipeError, EOFError, OSError):
                # Ignore errors caused by a closed or broken pipe, as it indicates
                # the pipe is no longer available for reading
                pass
            except KeyboardInterrupt:
                # Handle manual interruption (e.g., Ctrl+C) gracefully
                logging.info("KeyboardInterrupt detected, exiting stdout monitoring...")
                break
            except Exception as e:
                logging.error(f"Unexpected error while reading from stdout: {e}")
                logging.error(traceback.format_exc())
                break
            time.sleep(0.1)

    def _transcription_worker(*args, **kwargs):
        worker = TranscriptionWorker(*args, **kwargs)
        worker.run()

    def _recording_worker(self):
        """
        The worker method that continuously monitors audio input for voice activity.

        This method actively listens for voice activity and automatically starts or
        stops recording based on the detected activity. It also manages scenarios
        where buffer overflow may occur, ensuring stable performance during continuous
        monitoring and recording.
        """

        last_processing_check_time = 0
        last_buffer_message_time = 0
        previously_recording = False
        self.allow_early_transcription = True
        try:
            # Main loop to continuously monitor audio
            while self.transcriber_active:
                # Monitor processing time
                if last_processing_check_time:
                    last_processing_time = time.time() - last_processing_check_time
                    if last_processing_time > 0.1:
                        logging.warning("WARNING!: Processing took too long!")
                last_processing_check_time = time.time()
                try:
                    try:
                        data = self.audio_queue.get(timeout=0.01)
                    except queue.Empty:
                        # If the queue is empty, check if the worker should continue
                        if not self.transcriber_active:
                            break
                        continue  # If still running, continue to the next iteration

                    # Process recorded chunks if a callback is provided
                    if self.on_recorded_chunk:
                        self.on_recorded_chunk(data)

                    # Handle buffer overflow if enabled
                    if self.handle_buffer_overflow and self.audio_queue.qsize() > self.max_allowed_latency:
                        logging.warning(f"Audio queue size ({self.audio_queue.qsize()}) exceeds the latency limit. "
                                        f"Discarding old audio chunks.")
                        while self.audio_queue.qsize() > self.max_allowed_latency:
                            data = self.audio_queue.get()
                except BrokenPipeError:
                    logging.error("BrokenPipeError _recording_worker")
                    self.transcriber_active = False
                    break

                # Monitor the time since the last buffer message for controlling intervals
                if last_buffer_message_time:
                    time_passed = time.time() - last_buffer_message_time
                    if time_passed > 1:
                        last_buffer_message_time = time.time()
                else:
                    last_buffer_message_time = time.time()

                stop_attempt_failed = False
                if not (self.recording_active or self.recording_end_timestamp):
                    # Set appropriate state based on recording status
                    self._update_state(self.State.LISTENING if self.listen_start else self.State.INACTIVE)

                    # Detect voice activity to trigger recording
                    if self.auto_start_on_voice:
                        # If voice activity is detected, start recording
                        if self._is_voice_active():
                            logging.info("voice activity detected")
                            self.start()

                            self.auto_start_on_voice = False

                            # Add buffered audio to recording frames
                            self.frames.extend(list(self.audio_buffer))
                            self.audio_buffer.clear()

                            self.silero_vad_model.reset_states()
                        else:
                            # If no voice activity, continue checking
                            copied_data = data[:]
                            self._check_voice_activity(copied_data)

                    # Set silence start time to ensure it's ready for the next detection
                    self.silence_start_time_after_speech = 0

                # Handle active recording state: stopping on silence, etc.
                else:
                    # Stop recording when silence is detected following a period of speech
                    if self.auto_stop_on_silence:
                        is_speech = (
                            self._detect_speech_silero(data) if self.silero_inactivity_detection
                            else self._detect_speech_webrtc(data, True)
                        )

                        if not is_speech:
                            # Begin tracking silence duration before stopping the recording if no speech is detected
                            if self.silence_start_time_after_speech == 0 and \
                                    (time.time() - self.recording_start_time > self.min_recording_duration):
                                self.silence_start_time_after_speech = time.time()

                            # Determine if early transcription should be triggered
                            # based on the duration of detected silence
                            if self.silence_start_time_after_speech and self.silence_to_trigger_transcription and len(
                                    self.frames) > 0 and \
                                    (time.time() - self.silence_start_time_after_speech >
                                     self.silence_to_trigger_transcription) and \
                                    self.allow_early_transcription:
                                # Convert recorded frames to audio array and prepare for early transcription
                                self.transcribe_count += 1
                                audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
                                audio = audio_array.astype(np.float32) / config.INT16_MAX_ABS_VALUE

                                self.parent_transcription_pipe.send((audio, self.language))
                                self.allow_early_transcription = False
                        else:
                            if self.silence_start_time_after_speech:
                                self.silence_start_time_after_speech = 0
                                self.allow_early_transcription = True

                        # Wait for a period of silence to conclude recording after speech is detected
                        if self.silence_start_time_after_speech and time.time() - \
                                self.silence_start_time_after_speech >= \
                                self.post_speech_silence_time:

                            self.frames.append(data)
                            self.stop()
                            if not self.recording_active:
                                self.silence_start_time_after_speech = 0

                                self.listen_start = time.time()
                                self._update_state(self.State.LISTENING)
                                self.auto_start_on_voice = True
                            else:
                                stop_attempt_failed = True

                if not self.recording_active and previously_recording:
                    # Reset the system to a clean state after stopping the recording
                    self.auto_stop_on_silence = False

                if time.time() - self.silero_check_time > 0.1:
                    self.silero_check_time = 0

                previously_recording = self.recording_active

                if self.recording_active and not stop_attempt_failed:
                    self.frames.append(data)

                if not self.recording_active or self.silence_start_time_after_speech:
                    self.audio_buffer.append(data)
        except Exception as e:
            logging.debug('Debug: Caught exception in main try block')
            if not self.interrupt_stop_event.is_set():
                logging.error(f"Unhandled exception in _recording_worker: {e}")
                raise

    def _realtime_worker(self):
        """
        Performs real-time transcription if the feature is enabled.

        The method handles the process of transcribing recorded audio frames
        in real-time based on a specified resolution interval. The transcribed
        text is stored in `self.realtime_transcription_text`, and a callback
        function is invoked with this text if specified. It also manages
        transcription stabilization to ensure accurate, consistent output.
        """

        try:
            logging.info('Starting real-time worker')

            # Return immediately if real-time transcription is not enabled
            if not self.enable_realtime_transcription:
                return

            # Continue running as long as the main process is active
            while self.transcriber_active:
                # Check if the recording is active
                if self.recording_active:
                    # Pause processing to align with the real-time resolution interval
                    time.sleep(self.realtime_processing_pause)

                    # Convert recorded frames (byte data) into a NumPy array
                    audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)

                    logging.info(f"Current real-time buffer size: {len(audio_array)}")

                    # Normalize the audio array to a range of [-1, 1]
                    audio_array = audio_array.astype(np.float32) / config.INT16_MAX_ABS_VALUE

                    segments, info = self.realtime_model_type.transcribe(
                        audio_array,
                        task=self.task,
                        language=self.language if self.language else None,
                        beam_size=self.beam_size_realtime,
                        initial_prompt=self.initial_prompt,
                        suppress_tokens=self.suppress_tokens,
                    )

                    self.detected_realtime_language = info.language if info.language_probability > 0 else None
                    self.detected_realtime_language_probability = info.language_probability
                    realtime_text = " ".join(seg.text for seg in segments)
                    logging.info(f"Real-time text detected: {realtime_text}")

                    # Double-check recording status because it could change mid-transcription
                    if self.recording_active and time.time() - self.recording_start_time > 0.5:
                        self.realtime_transcription_text = realtime_text.strip()

                        self.text_storage.append(self.realtime_transcription_text)

                        # If there are at least two previous transcriptions, attempt to stabilize
                        if len(self.text_storage) >= 2:
                            last_two_texts = self.text_storage[-2:]

                            # Find the longest common prefix between the last two texts
                            prefix = os.path.commonprefix([last_two_texts[0], last_two_texts[1]])

                            # Use the prefix as the "safely detected" text if it is longer than the previous
                            if len(prefix) >= len(self.realtime_stabilized_safetext):
                                self.realtime_stabilized_safetext = prefix

                        # Attempt to match the stabilized text with the new transcription
                        matching_pos = find_suffix_match_in_text(
                            self.realtime_stabilized_safetext,
                            self.realtime_transcription_text
                        )

                        if matching_pos < 0:
                            # No match found, use the stabilized or current transcription
                            self._notify_realtime_transcription_stabilized(
                                self._preprocess_text_output(
                                    self.realtime_stabilized_safetext or self.realtime_transcription_text,
                                    True
                                )
                            )
                        else:
                            # Identified segments of the stabilized text within the transcribed text
                            # Merge the stabilized text with the newly transcribed segment
                            output_text = self.realtime_stabilized_safetext + \
                                          self.realtime_transcription_text[matching_pos:]

                            # This provides the "left" segment as stabilized text
                            # while simultaneously delivering newly detected parts
                            # on the first run, eliminating the need for two transcriptions
                            self._notify_realtime_transcription_stabilized(
                                self._preprocess_text_output(output_text, True)
                            )

                        # Execute the callback function with the transcribed text
                        self._notify_realtime_transcription_update(
                            self._preprocess_text_output(
                                self.realtime_transcription_text,
                                True
                            )
                        )
                # If not currently recording, pause briefly before the next check
                else:
                    time.sleep(config.PROCESS_SLEEP_INTERVAL)

        except Exception as e:
            logging.error(f"Unhandled exception in _realtime_worker: {e}")
            raise

    def wait_audio(self):
        """
        Waits for the start and completion of the audio recording process.

        This method is responsible for:
        - Waiting for voice activity to begin recording if not yet started.
        - Waiting for voice inactivity to complete the recording.
        - Setting the audio buffer from the recorded frames.
        - Resetting recording-related attributes.

        Side effects:
        - Updates the state of the instance.
        - Modifies the audio attribute to contain the processed audio data.
        """

        try:
            logging.info("Setting listen time")
            if self.listen_start == 0:
                self.listen_start = time.time()

            # If not yet started recording, wait for voice activity to initiate.
            if not self.recording_active and not self.frames:
                self._update_state(self.State.LISTENING)
                self.auto_start_on_voice = True

                # Wait until recording starts
                while not self.interrupt_stop_event.is_set():
                    if self.start_recording_event.wait(timeout=0.02):
                        break

            # If recording is ongoing, wait for voice inactivity
            # to finish recording.
            if self.recording_active:
                self.auto_stop_on_silence = True

                # Wait until recording stops
                logging.debug('Waiting for recording stop')
                while not self.interrupt_stop_event.is_set():
                    if self.stop_recording_event.wait(timeout=0.02):
                        break

            # Convert recorded frames to the appropriate audio format.
            audio_array = np.frombuffer(b''.join(self.frames), dtype=np.int16)
            self.audio = audio_array.astype(np.float32) / config.INT16_MAX_ABS_VALUE
            self.frames.clear()

            # Reset recording-related timestamps
            self.recording_end_timestamp = 0
            self.listen_start = 0

            self._update_state(self.State.INACTIVE)

        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt in wait_audio, shutting down")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

    def transcribe(self):
        """
        Transcribes audio captured by this class instance using the `faster_whisper` model.

        Automatically starts recording when voice activity is detected if not manually
        started using `recorder.start()`. Automatically stops recording when no voice
        activity is detected if not manually stopped with `recorder.stop()`. Processes
        the recorded audio to generate a transcription.

        Returns:
            str: The transcription of the recorded audio (if no callback is provided).

        Raises:
            Exception: If an error occurs during the transcription process.
        """
        self._update_state(self.State.TRANSCRIBING)
        audio_copy = copy.deepcopy(self.audio)

        with self.transcription_lock:
            try:
                if self.transcribe_count == 0:
                    logging.debug("Adding transcription request, no early transcription started")
                    start_time = time.time()  # Start timing
                    self.parent_transcription_pipe.send((self.audio, self.language))
                    self.transcribe_count += 1

                while self.transcribe_count > 0:
                    logging.debug(
                        f"Receive from parent_transcription_pipe after sending transcription request, transcribe_count: {self.transcribe_count}")
                    status, result = self.parent_transcription_pipe.recv()
                    self.transcribe_count -= 1

                self.allow_early_transcription = True
                self._update_state(self.State.INACTIVE)
                if status == 'success':
                    segments, info = result
                    self.detected_language = info.language if info.language_probability > 0 else None
                    self.detected_language_probability = info.language_probability
                    self.last_transcription_bytes = audio_copy
                    transcription = self._preprocess_text_output(segments)
                    return transcription
                else:
                    logging.error(f"Transcription error: {result}")
                    raise Exception(result)
            except Exception as e:
                logging.error(f"Error during transcription: {str(e)}")
                raise e

    def text(self, on_transcription_finished=None):
        """
        Transcribes audio captured by this class instance using the `faster_whisper` model.

        - Automatically starts recording upon detecting voice activity if not manually
          started using `recorder.start()`.
        - Automatically stops recording upon detecting voice inactivity if not manually
          stopped with `recorder.stop()`.
        - Processes the recorded audio to generate transcription.

        Args:
            on_transcription_finished (callable, optional): A callback function
              to be executed when transcription is complete.
              If provided, transcription will be performed asynchronously, and
              the callback will receive the transcription as its argument.
              If omitted, the transcription will be performed synchronously, and
              the result will be returned directly.

        Returns:
            str: The transcription of the recorded audio (if no callback is set).
        """
        self.interrupt_stop_event.clear()
        self.was_interrupted.clear()

        try:
            self.wait_audio()
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt in text() method")
            self.shutdown()
            raise  # Re-raise the exception after cleanup

        # Check if the process was interrupted or needs to be shut down
        if self.is_shut_down or self.interrupt_stop_event.is_set():
            if self.interrupt_stop_event.is_set():
                self.was_interrupted.set()
            return ""

        # Perform transcription asynchronously if a callback is provided
        if on_transcription_finished:
            threading.Thread(target=on_transcription_finished,
                             args=(self.transcribe(),)).start()
        else:
            # Perform transcription synchronously and return the result
            return self.transcribe()

    def start(self):
        """
        Starts recording audio directly without waiting for voice activity.
        """

        # Ensure a minimum interval between stopping and starting recordings
        if time.time() - self.recording_end_timestamp < self.min_recording_gap:
            logging.info("Attempted to start recording too soon after stopping.")
            return self

        logging.info("recording started")
        self._update_state(self.State.RECORDING)
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.frames = []
        self.recording_active = True
        self.recording_start_time = time.time()
        self.silero_voice_detected = False
        self.webrtc_voice_detected = False

        # Clear stop event and set start event
        self.stop_recording_event.clear()
        self.start_recording_event.set()

        # Trigger any callback for when recording starts
        if self.on_recording_start:
            self.on_recording_start()

        return self

    def stop(self):
        """
        Stops recording audio.
        """

        # Ensure a minimum interval between starting and stopping the recording
        if time.time() - self.recording_start_time < self.min_recording_duration:
            logging.info("Attempted to stop recording too soon after starting.")
            return self

        logging.info("recording stopped")
        self.recording_active = False
        self.recording_end_timestamp = time.time()
        self.silero_voice_detected = False
        self.webrtc_voice_detected = False
        self.silero_check_time = 0

        # Clear start event and set stop event
        self.start_recording_event.clear()
        self.stop_recording_event.set()

        # Trigger any callback for when recording stops
        if self.on_recording_stop:
            self.on_recording_stop()

        return self

    def process_audio_chunk(self, chunk, original_sample_rate=16000):
        """
        Process an audio chunk and accumulate it until the buffer size is reached.
        Once enough data is accumulated, send it to the audio queue for processing.

        Args:
            chunk (bytes or np.ndarray): The incoming audio data. If it's a NumPy array, it will be processed.
            original_sample_rate (int): The sample rate of the audio chunk. Default is 16000 Hz.
        """

        # Initialize the buffer if it hasn't been set up yet
        if not hasattr(self, 'buffer'):
            self.buffer = bytearray()

        # If the input is a NumPy array, process it before adding to the buffer
        if isinstance(chunk, np.ndarray):
            # Convert stereo audio (two channels) to mono by averaging, if necessary
            if chunk.ndim == 2:
                chunk = np.mean(chunk, axis=1)

            # Resample the audio to 16kHz if it has a different sample rate
            if original_sample_rate != 16000:
                num_samples = int(len(chunk) * 16000 / original_sample_rate)
                chunk = resample(chunk, num_samples)

            # Ensure the audio data is in int16 format
            chunk = chunk.astype(np.int16)

            # Convert the NumPy array to bytes
            chunk = chunk.tobytes()

        # Append the chunk to the buffer
        self.buffer += chunk
        # Define the buffer size limit to prevent issues with short buffers (Silero complains if too short)
        buffer_limit = 2 * self.buffer_size

        # If the buffer has enough accumulated data, send it to the audio queue
        while len(self.buffer) >= buffer_limit:
            # Extract a segment of data from the buffer that matches the required buffer size
            data_to_process = self.buffer[:buffer_limit]

            # Remove the extracted data from the buffer
            self.buffer = self.buffer[buffer_limit:]

            # Place the processed data into the audio queue for further handling
            self.audio_queue.put(data_to_process)

    def shutdown(self):
        """
        Safely shuts down the audio recording system by stopping the
        recording worker, closing the audio stream, and terminating
        related processes and threads.
        """

        with self.shutdown_lock:
            if self.is_shut_down:
                return

            print("\033[91mReal-Time Speech to Text shutting down\033[0m")

            # Set flags to force `wait_audio()` and `text()` to exit
            self.is_shut_down = True
            self.start_recording_event.set()
            self.stop_recording_event.set()

            # Signal that the shutdown process has started
            self.shutdown_event.set()
            self.recording_active = False
            self.transcriber_active = False

            logging.info('Finishing recording thread')
            if self.recording_thread:
                self.recording_thread.join()

            logging.info('Terminating transcription process')
            self.transcript_process.join(timeout=10)
            if self.transcript_process.is_alive():
                logging.warning("Transcript process did not terminate in time. Terminating forcefully.")
                self.transcript_process.terminate()

            self.parent_transcription_pipe.close()

            logging.debug('Finishing real-time thread')
            if self.realtime_thread:
                self.realtime_thread.join()

            if self.enable_realtime_transcription and self.realtime_model_type:
                del self.realtime_model_type
                self.realtime_model_type = None
            # Run garbage collection to clean up any unused resources
            gc.collect()

    def _detect_speech_silero(self, audio_chunk):
        """
        Determines if speech is detected in the provided audio data using the Silero Voice Activity Detection (VAD) model.

        Args:
            audio_chunk (bytes): Raw bytes of audio data (e.g., 1024 bytes, assumed to be 16000 Hz sample rate, 16-bit per sample).

        Returns:
            bool: True if speech is detected based on the sensitivity threshold of the Silero model, otherwise False.
        """

        # Resample audio to 16000 Hz if the current sample rate is different
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(audio_chunk, dtype=np.int16)
            # Resample the audio to match 16000 Hz
            data_16000 = signal.resample_poly(pcm_data, 16000, self.sample_rate)
            audio_chunk = data_16000.astype(np.int16).tobytes()

        # Indicate that the Silero model is currently processing
        self.silero_working = True

        # Convert raw audio bytes to a NumPy array and normalize to a [-1, 1] range
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / config.INT16_MAX_ABS_VALUE

        # Use the Silero VAD model to get the voice activity probability
        vad_prob = self.silero_vad_model(
            torch.from_numpy(audio_array),
            self.sample_rate
        ).item()

        # Determine if speech is active based on the sensitivity threshold
        is_speech_detected = vad_prob > (1 - self.silero_sensitivity)

        # Update the state to reflect if speech is detected
        if is_speech_detected:
            self.silero_voice_detected = True

        # Reset Silero working state
        self.silero_working = False

        return is_speech_detected

    def _detect_speech_webrtc(self, audio_chunk, require_all_frames=False):
        """
        Determines if speech is detected in the provided audio data using WebRTC Voice Activity Detection (VAD).

        Args:
            audio_chunk (bytes): Raw bytes of audio data (1024 bytes, assumed to be 16000 Hz sample rate, 16-bit per sample).
            require_all_frames (bool): If True, requires all frames in the chunk to be classified as speech for the method to return True.
                                       If False, the method will return True as soon as any frame is classified as speech.

        Returns:
            bool: True if speech is detected based on the criteria set by `require_all_frames`, otherwise False.
        """

        # Resample audio to 16000 Hz if the sample rate is different
        if self.sample_rate != 16000:
            pcm_data = np.frombuffer(audio_chunk, dtype=np.int16)
            data_16000 = signal.resample_poly(pcm_data, 16000, self.sample_rate)
            audio_chunk = data_16000.astype(np.int16).tobytes()

        # Define frame length for 10ms (160 samples for 16000 Hz audio)
        frame_length = int(16000 * 0.01)  # 10ms frame length
        # Determine the total number of frames in the audio chunk
        num_frames = int(len(audio_chunk) / (2 * frame_length))  # 2 bytes per sample (16-bit)
        speech_frames_count = 0  # Counter to track the number of frames detected as speech

        # Iterate over each frame within the audio chunk
        for i in range(num_frames):
            # Calculate the start and end positions for each 10ms frame
            start_byte = i * frame_length * 2
            end_byte = start_byte + frame_length * 2
            frame = audio_chunk[start_byte:end_byte]

            # Use the WebRTC VAD model to check if the frame contains speech
            if self.webrtc_vad_model.is_speech(frame, 16000):
                speech_frames_count += 1  # Increment if speech is detected in the frame
                if not require_all_frames:
                    # If only one speech frame is enough, return True immediately
                    return True

        # If all frames must be true, determine if every frame was detected as speech
        if require_all_frames:
            # Return True only if every frame is classified as speech
            return speech_frames_count == num_frames
        else:
            # If not all frames must be speech, return False since no speech was detected
            return False

    def _check_voice_activity(self, audio_chunk):
        """
        Check if there is voice activity in the provided audio data.

        Args:
            audio_chunk (bytes): The audio data to be analyzed for voice activity.
        """
        self.webrtc_voice_detected = self._detect_speech_webrtc(audio_chunk)

        # Perform an initial voice activity check using WebRTC VAD
        if self.webrtc_voice_detected and not self.silero_working:
            self.silero_working = True

            # Proceed with further verification using Silero if WebRTC detects voice
            threading.Thread(target=self._detect_speech_silero, args=(audio_chunk,)).start()

    def clear_audio_queue(self):
        """
         Safely empties the audio queue and buffer to prevent processing of leftover audio fragments.
        """
        self.audio_buffer.clear()
        try:
            while True:
                self.audio_queue.get_nowait()
        except Exception as e:
            # PyTorch's mp.Queue does not have a specific Empty exception, so we catch any exception
            # that occurs when attempting to read from an empty queue and ignore it
            logging.error(e)
            pass

    def _is_voice_active(self):
        """
        Check if voice activity is currently detected.

        Returns:
            bool: True if both WebRTC and Silero models detect voice activity, False otherwise.
        """
        return self.webrtc_voice_detected and self.silero_voice_detected

    def _update_state(self, new_state):
        """
        Update the current state of the recorder and execute
        corresponding state-change callbacks.

        Args:
            new_state (State): The new state to set.
        """
        # Exit if the state has not changed
        if new_state == self.state:
            return

        # Store the current state for later comparison and update to the new state
        old_state = self.state
        self.state = new_state

        logging.info(f"State changed from '{old_state.value}' to '{new_state.value}'")
        # Optional: Add callback functions to handle state transitions

    def _preprocess_text_output(self, text, preview=False):
        """
        Preprocess the output text by removing excess whitespace, ensuring the first
        letter is capitalized, and optionally adding a period at the end.

        Args:
            text (str): The text to preprocess.
            preview (bool): If True, skips adding punctuation at the end. Default is False.

        Returns:
            str: The preprocessed text.
        """
        # Normalize whitespace and trim leading/trailing spaces
        text = re.sub(r'\s+', ' ', text.strip())

        # Capitalize the first letter if the feature is enabled
        if text:
            text = text[0].upper() + text[1:]

        # Add a period at the end if not in preview mode and if punctuation is needed
        if not preview and self.add_period_to_sentence:
            if text and text[-1].isalnum():
                text += '.'

        return text

    def _notify_realtime_transcription_stabilized(self, stabilized_text):
        """
        Notify external listeners when the real-time transcription stabilizes.

        This method is called internally when the transcription text is considered
        "stable," meaning it is less likely to change significantly with additional
        audio input. If recording is still ongoing, it invokes the callback to notify
        any registered external listener about the stabilized text. This is useful
        for applications that display live transcription results and want to highlight
        parts of the transcription that are less likely to change.

        Args:
            stabilized_text (str): The stabilized transcription text.
        """
        if self.on_realtime_transcription_stabilized:
            if self.recording_active:
                self.on_realtime_transcription_stabilized(stabilized_text)

    def _notify_realtime_transcription_update(self, updated_text):
        """
        Notify external listeners upon receiving an update in the real-time transcription.

        This method is called whenever there is a change in the transcription text,
        and if recording is still ongoing, it triggers the callback to notify any
        registered external listener about the update. This allows applications to
        receive and potentially display live transcription updates, which may be
        partial and subject to change.

        Args:
            updated_text (str): The updated transcription text.
        """
        if self.recording_active and self.on_realtime_transcription_update:
            self.on_realtime_transcription_update(updated_text)
