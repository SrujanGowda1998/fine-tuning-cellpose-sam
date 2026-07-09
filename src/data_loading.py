"""Load train/test data via the Cellpose loader and log shapes."""

import logging

from cellpose import io

logger = logging.getLogger(__name__)


def load_data(train_2d_dir, test_2d_dir, mask_filter="_masks"):
    """Return (train_data, train_labels, train_files, test_data, test_labels, test_files)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Loading training data")
    logger.info("=" * 70)

    output = io.load_train_test_data(
        str(train_2d_dir),
        str(test_2d_dir),
        mask_filter=mask_filter,
    )

    logger.info(f"Type: {type(output)}")
    try:
        logger.info(f"Length: {len(output)}")
    except Exception:
        logger.info("Output has no length")

    for i, item in enumerate(output):
        logger.info(f"Item {i}: type={type(item)}")

    (
        train_data,
        train_labels,
        train_files,
        test_data,
        test_labels,
        test_files,
    ) = output

    logger.info(f"Training images loaded: {len(train_data)}")
    logger.info(f"Testing images loaded : {len(test_data)}")

    return (
        train_data,
        train_labels,
        train_files,
        test_data,
        test_labels,
        test_files,
    )


def log_data_shapes(train_data, train_labels):
    """Debug helper: log per-image shapes and dtypes."""
    for i, img in enumerate(train_data):
        logger.info(f"TRAIN IMAGE {i}: shape={img.shape}, dtype={img.dtype}")

    for i, mask in enumerate(train_labels):
        logger.info(f"TRAIN MASK {i}: shape={mask.shape}, dtype={mask.dtype}")
