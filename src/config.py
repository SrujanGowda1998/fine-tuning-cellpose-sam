"""Paths, hyperparameters, and a single run timestamp shared across the pipeline."""

from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_DIR = Path("/g/koehler/Srujan/cellpose/fine-tuning-cellpose-sam")

TRAIN_DIR = PROJECT_DIR / "data/train1"
TEST_DIR = PROJECT_DIR / "data/test"

# Cellpose-SAM trains in 2D, so 3D stacks are sliced into these dirs
TRAIN_2D_DIR = PROJECT_DIR / "data/train_2d"
TEST_2D_DIR = PROJECT_DIR / "data/test_2d"

MODEL_DIR = PROJECT_DIR / "models"
LOG_DIR = PROJECT_DIR / "logs"
PRED_DIR = PROJECT_DIR / "predictions"

# One timestamp per run, reused for the log file and the model name
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ------------------------------------------------------------------
# Dataset / training hyperparameters
# ------------------------------------------------------------------
MASK_FILTER = "_masks"
PRETRAINED_MODEL = "cpsam_v2"  # Other available models "cpdino", "cpdino-vitb", "cpsam"

N_EPOCHS = 6
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
BATCH_SIZE = 1

# derived
LOG_FILE = LOG_DIR / f"training_{TIMESTAMP}.log"
MODEL_NAME = MODEL_DIR / f"cellpose_sam_finetuned_{TIMESTAMP}"
LOSS_CURVE_FILE = LOG_DIR / f"loss_curve_{TIMESTAMP}.png"


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (MODEL_DIR, LOG_DIR, PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)
