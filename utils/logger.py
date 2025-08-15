import logging
import os
from datetime import datetime

# Generate log filename
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Just the directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Full path to log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Setup logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d - %(funcName)s()] - %(message)s",
    level=logging.INFO
)

# Create logger object
logger = logging.getLogger(__name__)