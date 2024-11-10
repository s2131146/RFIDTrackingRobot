import logging
import os

LOG_FILENAME = "tracker.log"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if os.path.exists(LOG_FILENAME):
    os.remove(LOG_FILENAME)

file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "[Tracker %(asctime)s.%(msecs)03d] %(levelname)s: %(message)s"
)
formatter.datefmt = "%H:%M:%S"
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
