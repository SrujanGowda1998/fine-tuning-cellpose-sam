#!/usr/bin/env python3
"""
Fine-tune Cellpose-SAM (Cellpose 4.2.1.1).

Run from this directory:
    python main.py

Dataset format:
    data/train/  data/test/
        image1.tif
        image1_masks.tif   (0=background, 1,2,3...=instances)
    3D TIFF stacks (Z,Y,X), uint16. Sliced into 2D pairs before training.
"""

import config
from logging_setup import setup_logging
from gpu import get_device
from data_prep import validate_dataset, slice_volume_dir
from data_loading import load_data, log_data_shapes
from training import init_model, train_model
from evaluation import (
    load_finetuned_model,
    run_inference,
    compute_average_precision,
    save_predictions,
)
from plotting import save_loss_curve


def main():
    config.ensure_dirs()
    logger = setup_logging(config.LOG_FILE)

    logger.info("=" * 70)
    logger.info("Cellpose-SAM fine tuning started")
    logger.info("=" * 70)

    # 1. Device
    get_device()

    # 2. Validate + slice 3D stacks into 2D pairs
    validate_dataset(config.TRAIN_DIR, mask_filter=config.MASK_FILTER)
    validate_dataset(config.TEST_DIR, mask_filter=config.MASK_FILTER)
    slice_volume_dir(config.TRAIN_DIR, config.TRAIN_2D_DIR, mask_filter=config.MASK_FILTER)
    slice_volume_dir(config.TEST_DIR, config.TEST_2D_DIR, mask_filter=config.MASK_FILTER)

    # 3. Load data
    (
        train_data,
        train_labels,
        train_files,
        test_data,
        test_labels,
        test_files,
    ) = load_data(config.TRAIN_2D_DIR, config.TEST_2D_DIR, mask_filter=config.MASK_FILTER)
    log_data_shapes(train_data, train_labels)

    # 4. Model + training
    model = init_model(config.PRETRAINED_MODEL)
    new_model_path, train_losses, test_losses = train_model(
        model,
        train_data,
        train_labels,
        test_data,
        test_labels,
        model_dir=config.MODEL_DIR,
        model_name=config.MODEL_NAME,
        n_epochs=config.N_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        batch_size=config.BATCH_SIZE,
    )

    # 5. Evaluation
    finetuned_model = load_finetuned_model(new_model_path)
    predicted_masks = run_inference(finetuned_model, test_data)
    mean_ap = compute_average_precision(test_labels, predicted_masks)
    save_predictions(predicted_masks, config.PRED_DIR)

    # 6. Loss curves
    save_loss_curve(train_losses, test_losses, config.LOSS_CURVE_FILE)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Final model: {new_model_path}")
    logger.info(f"Mean AP@0.5: {mean_ap:.4f}")


if __name__ == "__main__":
    main()
