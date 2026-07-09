"""GPU / device detection."""

import logging

import torch

logger = logging.getLogger(__name__)


def get_device():
    """Return a torch.device, logging what was detected."""
    logger.info("Checking CUDA availability...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.warning("No CUDA GPU detected. Training will run on CPU.")

    logger.info(f"Using device: {device}")
    return device
