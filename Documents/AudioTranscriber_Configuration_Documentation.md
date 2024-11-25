# AudioTranscriber Configuration Documentation

This document provides an overview of the configuration variables used in the `AudioTranscriber` class. Each variable is explained, along with its purpose and default value.

## Table of Contents
1. [Model Configuration](#model-configuration)
2. [Audio Input Settings](#audio-input-settings)
3. [Recording Control](#recording-control)
4. [Realtime Transcription Settings](#realtime-transcription-settings)
5. [Voice Activation Detection (VAD) Settings](#voice-activation-detection-vad-settings)
6. [Audio Buffer Settings](#audio-buffer-settings)
7. [General Settings](#general-settings)

## Model Configuration

| Variable              | Description                                                                                                                                                                                                                                                | Default Value |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `model`               | The transcription model type used for the refined processing.                                                                                                                                                                                              | `config.MODEL_TRANSCRIPTION_TYPE` |
| `compute_type`        | Specifies the compute configuration for the Faster Whisper model. This could be set to `"default"`, `"int8"`, `"float16"`, or `"float32"` depending on the desired precision and performance. Lower precision can improve speed but might affect accuracy. | `"default"` |
| `gpu_device_index`    | Index of the GPU device to use. Can be a single integer or a list of integers.                                                                                                                                                                             | `0` |
| `beam_size`           | Beam size for the transcription algorithm. Larger values may improve accuracy but increase latency.                                                                                                                                                        | `5` |
| `realtime_model_type` | Type of model used for real-time transcription.                                                                                                                                                                                                            | `config.MODEL_REALTIME_TRANSCRIPTION_TYPE` |
| `language` | Specifies the language for transcription. If left as an empty string, the model will attempt to automatically detect the language. Setting this parameter can improve accuracy and speed if the language is known beforehand. | `""` |
| `task` | Defines the specific task the transcription model should perform. Typical options are `"transcribe"` (to convert speech to text) or `"translate"` (to convert speech to text in a different language). This allows the model to be used for transcription or translation purposes, depending on the use case. | `config.MODEL_TASK_TYPE` |

## Audio Input Settings

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `input_device_index` | Index of the audio input device to use for recording. | `None` |
| `sample_rate` | The sample rate of the audio input. | `config.AUDIO_SAMPLE_RATE` |

## Recording Control

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `on_recording_start` | Callback function triggered when recording starts. | `None` |
| `on_recording_stop` | Callback function triggered when recording stops. | `None` |
| `min_recording_duration` | Minimum duration (in seconds) for a recording to be considered valid. | `config.MIN_RECORDING_DURATION` |
| `min_recording_gap` | Minimum gap (in seconds) between recordings. | `config.MIN_RECORDING_GAP` |
| `pre_recording_buffer_time` | Duration (in seconds) of audio buffered before recording starts. | `config.PRE_RECORDING_BUFFER_TIME` |
| `post_speech_silence_time` | Duration (in seconds) of silence after speech before considering the recording complete. | `config.POST_SPEECH_SILENCE_TIME` |

## Realtime Transcription Settings

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `enable_realtime_transcription` | Enables real-time transcription processing. | `False` |
| `realtime_processing_pause` | Time interval (in seconds) to pause between real-time transcription updates. | `config.REALTIME_PROCESSING_PAUSE` |
| `on_realtime_transcription_update` | Callback function for real-time transcription updates. | `None` |
| `on_realtime_transcription_stabilized` | Callback function for stabilized real-time transcription. | `None` |
| `beam_size_realtime` | Beam size used specifically for real-time transcription. | `3` |

## Voice Activation Detection (VAD) Settings

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `silero_sensitivity` | Sensitivity level for Silero VAD. | `config.SILERO_VAD_SENSITIVITY` |
| `silero_use_onnx` | Determines if ONNX should be used for Silero VAD. | `True` |
| `silero_inactivity_detection` | Enables inactivity detection for Silero VAD. | `False` |
| `webrtc_sensitivity` | Sensitivity level for WebRTC VAD. | `config.WEBRTC_VAD_SENSITIVITY` |
| `on_vad_detect_start` | Callback function triggered when VAD detects the start of speech. | `None` |
| `on_vad_detect_stop` | Callback function triggered when VAD detects the end of speech. | `None` |

## Audio Buffer Settings

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `buffer_size` | Size of the audio buffer. | `config.AUDIO_BUFFER_SIZE` |
| `handle_buffer_overflow` | Determines if the buffer overflow should be handled automatically. | `config.HANDLE_BUFFER_OVERFLOW` |
| `audio_queue` | Queue for managing audio data to be processed. | `mp.Queue()` |

## General Settings

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `add_period_to_sentence` | Determines whether a period (`.`) should be automatically added to the end of each transcribed sentence.  | `True` |
| `logging_level` | Logging level (e.g., DEBUG, INFO, WARNING). | `logging.INFO` |
| `initial_prompt` | Initial prompt used for the transcription model. Can be a string or an iterable of tokens. | `None` |
| `suppress_tokens` | List of tokens to suppress during transcription. | `[-1]` |
| `max_allowed_latency` | Maximum allowed latency for the transcription. | `config.MAX_ALLOWED_LATENCY` |
| `disable_log_file` | Disables logging to a file if set to `True`. | `True` |
| `early_transcription_on_silence` | The amount of silence needed to trigger early transcription. | `0` |
| `auto_start_on_voice` | Automatically starts recording when voice is detected. | `False` |
| `auto_stop_on_silence` | Automatically stops recording when silence is detected. | `False` |

## Initialization and Setup Functions

| Variable | Description |
|----------|-------------|
| `transcript_process` | Process that starts the audio transcriber worker. |
| `realtime_thread` | Thread that manages real-time transcription tasks. |
| `recording_thread` | Thread that manages audio recording tasks. |
| `audio_interface` | Interface used to manage the audio hardware. |
| `silero_vad_model` | Initialized model for Silero VAD. |
| `webrtc_vad_model` | Initialized model for WebRTC VAD. |

