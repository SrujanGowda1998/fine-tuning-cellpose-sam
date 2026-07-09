"""Logging configuration. Call setup_logging() once from main; every other
module just does `logger = logging.getLogger(__name__)`."""

import logging
import sys


def setup_logging(log_file):
    """Configure the root logger to write to both a file and stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)
