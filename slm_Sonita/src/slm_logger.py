import os
import shutil
import logging
import time
from datetime import datetime, timedelta

def setup_logging(log_filename):
    """Set up logging to a shared log file and console."""
    slm_dir = os.path.join(os.path.dirname(__file__), '..', 'slm_logs')
    
    # Remove and recreate the log directory
    # if os.path.exists(slm_dir):
        # shutil.rmtree(slm_dir)
    os.makedirs(slm_dir, exist_ok=True)

    # Full log file path inside slm_logs
    log_path = os.path.join(slm_dir, log_filename)

    # Configure root logger
    logger = logging.getLogger()
    logger.handlers.clear()  # Ensure no duplicate handlers
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def setup_data_logging():
    RETENTION_DAYS = 30
    
    """Set up data logging directory."""
    data_log_dir = os.path.join(os.path.dirname(__file__), '..', 'data_logs')
    os.makedirs(data_log_dir, exist_ok=True)
    
    cutoff = time.time() - (RETENTION_DAYS * 86400)

    for filename in os.listdir(data_log_dir):
        filepath = os.path.join(data_log_dir, filename)

        if os.path.isfile(filepath):
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
    
    return data_log_dir
