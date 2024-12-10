import os
# Environment Settings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import asyncio
import websockets
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import wave
from faster_whisper import WhisperModel
import torch

from config import *

# Initialize the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = WhisperModel(MODEL_NAME, device=device, compute_type="float16" if device == "cuda" else "float32")
    logger.info(f"Loaded Whisper model '{MODEL_NAME}' on device '{device}'.")
except Exception as e:
    logger.exception("Failed to load Whisper model.")
    raise e

# Semaphore to limit concurrent transcriptions
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# ThreadPoolExecutor for blocking operations
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TRANSCRIPTIONS)

async def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio data using Faster Whisper model."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        # Run the blocking transcription in a separate thread
        segments, _ = await loop.run_in_executor(
            executor,
            lambda: model.transcribe(BytesIO(audio_bytes), task="transcribe", language=LANGUAGE, beam_size=BEAM_SIZE)
        )
        transcription = " ".join(segment.text for segment in segments)
        return transcription

async def process_wav(websocket, path):
    client_ip = websocket.remote_address[0]
    logger.info(f"Connection opened from {client_ip}")
    
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio_bytes = message
                # Validate WAV format
                try:
                    with wave.open(BytesIO(audio_bytes), 'rb') as audio:
                        if audio.getnchannels() != 1:
                            error_msg = "Error: Audio must be mono."
                            await websocket.send(json.dumps({'error': error_msg}))
                            logger.warning(f"{client_ip} - {error_msg}")
                            continue
                        if audio.getsampwidth() not in [2, 3, 4]:
                            error_msg = "Error: Unsupported sample width."
                            await websocket.send(json.dumps({'error': error_msg}))
                            logger.warning(f"{client_ip} - {error_msg}")
                            continue
                        # Additional validations can be added here
                except wave.Error as e:
                    error_msg = f"Error: Invalid WAV file. {e}"
                    await websocket.send(json.dumps({'error': error_msg}))
                    logger.warning(f"{client_ip} - {error_msg}")
                    continue

                try:
                    transcription = await transcribe_audio(audio_bytes)
                    logger.info(f"{client_ip} - Transcription: {transcription}")
                    response = {'transcription': transcription}
                    await websocket.send(json.dumps(response))
                except Exception as e:
                    error_msg = f"Error during transcription: {str(e)}"
                    await websocket.send(json.dumps({'error': error_msg}))
                    logger.exception(f"{client_ip} - {error_msg}")
            else:
                error_msg = "Error: Received non-binary data."
                await websocket.send(json.dumps({'error': error_msg}))
                logger.warning(f"{client_ip} - {error_msg}")
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed from {client_ip}: {e.code} - {e.reason}")
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        try:
            await websocket.send(json.dumps({'error': error_msg}))
        except:
            pass  # If sending fails, ignore
        logger.exception(f"{client_ip} - {error_msg}")
    finally:
        logger.info(f"Connection with {client_ip} closed.")

async def main():
    server = await websockets.serve(
        process_wav,
        HOST,
        PORT,
        max_size=MAX_MESSAGE_SIZE,
        max_queue=100,  # Limit the number of queued messages
        ping_interval=20,
        ping_timeout=20,
    )
    logger.info(f"WebSocket server running on ws://{HOST}:{PORT}")
    try:
        await asyncio.Future()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        server.close()
        await server.wait_closed()
        executor.shutdown(wait=True)
        logger.info("WebSocket server shut down gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped manually.")
    except Exception as e:
        logger.exception("Server encountered an unexpected error.")