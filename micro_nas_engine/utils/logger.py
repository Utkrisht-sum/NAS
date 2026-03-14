import logging
import os
import sys

def setup_logger(name="micro_nas", log_level=logging.INFO):
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)

        # File handler
        file_handler = logging.FileHandler("logs/micro_nas.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

logger = setup_logger()
