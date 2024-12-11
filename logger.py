from datetime import datetime
import logging
import os


LOG_FILENAME = f"records\RFIDTR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
DEBUG_CONSOLE = False

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if os.path.exists(LOG_FILENAME):
    os.remove(LOG_FILENAME)

file_handler = logging.FileHandler(LOG_FILENAME)
file_handler.setLevel(logging.DEBUG)

if DEBUG_CONSOLE:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    "[Tracker %(asctime)s.%(msecs)03d] %(levelname)s: %(message)s"
)
formatter.datefmt = "%H:%M:%S"
file_handler.setFormatter(formatter)
if DEBUG_CONSOLE:
    console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
if DEBUG_CONSOLE:
    logger.addHandler(console_handler)
