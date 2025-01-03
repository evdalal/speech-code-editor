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

# config.py

import platform

# Model Configuration
MODEL_TRANSCRIPTION_TYPE = "large-v2"
MODEL_REALTIME_TRANSCRIPTION_TYPE = "base.en"
MODEL_TASK_TYPE = "transcribe"

# Real-Time Processing Settings
REALTIME_PROCESSING_PAUSE = 0.2
SILERO_VAD_SENSITIVITY = 0.4
WEBRTC_VAD_SENSITIVITY = 3
POST_SPEECH_SILENCE_TIME = 0.6

# Recording Control
MIN_RECORDING_DURATION = 0.5
MIN_RECORDING_GAP = 0
PRE_RECORDING_BUFFER_TIME = 1.0

# Buffer and Latency Management
MAX_ALLOWED_LATENCY = 100
PROCESS_SLEEP_INTERVAL = 0.02
AUDIO_SAMPLE_RATE = 16000
AUDIO_BUFFER_SIZE = 512

# General Audio Constants
INT16_MAX_ABS_VALUE = 32768.0


# Buffer Overflow Handling
HANDLE_BUFFER_OVERFLOW = platform.system() != 'Darwin'
