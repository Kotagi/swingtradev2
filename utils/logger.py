# src/utils/logger.py

"""
logger.py

Provides a standardized logger setup for both console and file output.
Modules can call `setup_logger` at startup to get a Logger that writes
to both stdout and a file with consistent formatting.
"""

import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = __name__,
    log_file: str = "pipeline.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a Logger instance.

    This logger will output messages to both the console (stdout) and a file.

    Args:
        name: Name of the logger (commonly __name__ of the calling module).
        log_file: Path to the log file where logs should be written. 
                  The directory will be created if it does not exist.
        level: Logging level threshold (e.g., logging.INFO, logging.DEBUG).

    Returns:
        A configured logging.Logger instance.
    """
    # Create or retrieve the named logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if already configured
    if not logger.handlers:
        # Define a consistent log message format
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )

        # Console handler: stream to stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Ensure the directory for the log file exists
        log_path = Path(log_file)
        if log_path.parent and not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler: append logs to the specified file
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

