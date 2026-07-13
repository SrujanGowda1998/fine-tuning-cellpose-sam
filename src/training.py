"""Model initialization and fine-tuning."""

import logging

import torch
from cellpose import models, train

logger = logging.getLogger(__name__)


def init_model(pretrained_model):
    """Load the pretrained Cellpose-SAM model."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Initializing Cellpose-SAM model")
    logger.info("=" * 70)

    model = models.CellposeModel(
        gpu=torch.cuda.is_available(),
        pretrained_model=pretrained_model,
    )

    logger.info("Cellpose-SAM model initialized")
    return model


def train_model(
    model,
    train_data,
    train_labels,
    test_data,
    test_labels,
    *,
    model_dir,
    model_name,
    n_epochs,
    learning_rate,
    weight_decay,
    batch_size,
):
    """Fine-tune the model and return (new_model_path, train_losses, test_losses)."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Training parameters")
    logger.info("=" * 70)
    logger.info(f"Epochs        : {n_epochs}")
    logger.info(f"Learning rate : {learning_rate}")
    logger.info(f"Weight decay  : {weight_decay}")
    logger.info(f"Batch size    : {batch_size}")
    logger.info(f"Model output  : {model_name}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Starting fine tuning")
    logger.info("=" * 70)

    try:
        new_model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            # for small datasets
            nimg_per_epoch=max(2, len(train_data)),
            nimg_test_per_epoch=max(1, len(test_data)),
            # save trained model
            save_path=str(model_dir),
            model_name=str(model_name),
            # normalize microscopy images
            normalize=True,
            # compute flows during training
            compute_flows=True,
            min_train_masks=1,
        )
    except Exception:
        logger.exception("Training failed")
        raise

    logger.info("")
    logger.info("=" * 70)
    logger.info("Training completed")
    logger.info("=" * 70)
    logger.info(f"Saved model path: {new_model_path}")

    return new_model_path, train_losses, test_losses
