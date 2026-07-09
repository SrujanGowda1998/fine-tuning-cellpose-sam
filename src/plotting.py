"""Plot and save the training/test loss curves."""

import logging

import matplotlib

matplotlib.use("Agg")  # headless (SLURM / no display)
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


def save_loss_curve(train_losses, test_losses, out_file):
    """Save a training/test loss curve to out_file."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Saving training history")
    logger.info("=" * 70)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="training loss")
    plt.plot(test_losses, label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Cellpose-SAM fine tuning")

    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Loss curve saved: {out_file}")
