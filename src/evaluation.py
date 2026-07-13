"""Evaluation: load the fine-tuned model, run inference, score, and save masks."""

import logging

import numpy as np
import tifffile
import torch
from cellpose import models, metrics

import matplotlib.pyplot as plt
from cellpose import plot

logger = logging.getLogger(__name__)


def load_finetuned_model(model_path):
    """Load a fine-tuned model from disk for evaluation."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Loading fine-tuned model for evaluation")
    logger.info("=" * 70)

    model = models.CellposeModel(
        gpu=torch.cuda.is_available(),
        pretrained_model=model_path,
    )

    logger.info("Fine-tuned model loaded successfully")
    return model


def run_inference(model, test_data):
    """Run the model on every test image, visualize predictions, and return predicted masks."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Running inference on test dataset")
    logger.info("=" * 70)

    predicted_masks = []

    for i, image in enumerate(test_data):
        logger.info(f"Predicting test image {i + 1}/{len(test_data)}")

        masks, flows, styles = model.eval(image, batch_size=1)
        predicted_masks.append(masks)

        logger.info(f"Prediction shape: {masks.shape}")
        logger.info(f"Detected objects: {len(np.unique(masks)) - 1}")

        # Visualize prediction
        fig = plt.figure(figsize=(10, 10))
        plot.show_segmentation(
            fig,
            image,
            masks,
            flows[0],
            channels=[0, 0]
        )
        plt.suptitle(f"Test Image {i + 1}")
        plt.show()

    return predicted_masks


def compute_average_precision(test_labels, predicted_masks):
    """Return mean average precision at IoU=0.5."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Calculating segmentation performance")
    logger.info("=" * 70)

    ap = metrics.average_precision(test_labels, predicted_masks)
    mean_ap = ap[:, 0].mean()

    logger.info(f"Average Precision IoU=0.5: {mean_ap:.4f}")
    return mean_ap


def save_predictions(predicted_masks, pred_dir):
    """Write each predicted mask to disk as a uint16 TIFF."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Saving predicted masks")
    logger.info("=" * 70)

    for i, pred in enumerate(predicted_masks):
        output_file = pred_dir / f"prediction_{i + 1}.tif"
        tifffile.imwrite(output_file, pred.astype(np.uint16))
        logger.info(f"Saved: {output_file}")
