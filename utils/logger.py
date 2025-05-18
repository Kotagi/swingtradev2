"""Logging configuration for feature pipeline and other modules."""
import logging
import sys

def setup_logger(name=__name__, log_file='pipeline.log', level=logging.INFO):
    """
    Set up and return a logger instance that logs to both console and file.
    - name: logger name (usually __name__)
    - log_file: path to the log file
    - level: logging level (e.g., logging.INFO)
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
