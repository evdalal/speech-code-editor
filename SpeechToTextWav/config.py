import logging

# Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 8080
MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_CONCURRENT_TRANSCRIPTIONS = 4  # Adjust based on server capacity
MODEL_NAME = "medium.en"
LANGUAGE = "en"
BEAM_SIZE = 5

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)