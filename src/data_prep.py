"""Dataset validation and 3D -> 2D slicing."""

import logging

import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def validate_dataset(directory, mask_filter="_masks"):
    """
    Validate a Cellpose TIFF dataset.

    Expected layout:
        directory/
            image1.tif
            image1_masks.tif
            ...

    Checks: image/mask exist, matching shapes, dtypes, instance labels.
    Masks that aren't uint16 are converted in place.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Validating dataset: {directory}")
    logger.info("=" * 70)

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    image_files = sorted(
        f for f in directory.glob("*.tif") if mask_filter not in f.name
    )

    if len(image_files) == 0:
        raise RuntimeError(f"No TIFF images found in {directory}")

    logger.info(f"Found {len(image_files)} images")

    for img_path in image_files:
        mask_path = directory / f"{img_path.stem}{mask_filter}.tif"

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path.name}")

        img = tifffile.imread(img_path)
        mask = tifffile.imread(mask_path)

        # Ensure instance labels are integers
        if mask.dtype != np.uint16:
            mask = mask.astype(np.uint16)
            tifffile.imwrite(mask_path, mask)
            logger.info(f"Converted mask dtype to uint16: {mask_path.name}")

        logger.info("-" * 60)
        logger.info(f"Image: {img_path.name}")
        logger.info(f" Image shape : {img.shape}")
        logger.info(f" Image dtype : {img.dtype}")
        logger.info(f" Mask shape  : {mask.shape}")
        logger.info(f" Mask dtype  : {mask.dtype}")

        if img.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch:\n{img_path.name}\n"
                f"Image {img.shape}\nMask  {mask.shape}"
            )

        unique_labels = np.unique(mask)
        logger.info(f"Unique labels: {unique_labels[:15]}")
        logger.info(f"Number of cells: {len(unique_labels) - 1}")

        if unique_labels[0] != 0:
            logger.warning("Mask does not contain background label 0")

        if img.dtype != np.uint16:
            logger.warning(f"Image is not uint16: {img.dtype}")

    logger.info("")
    logger.info("Dataset validation completed successfully")


def slice_volume_dir(src_dir, dst_dir, mask_filter="_masks", min_masks=1):
    """Convert 3D (Z,Y,X) tif pairs in src_dir into 2D per-slice pairs in dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in src_dir.glob("*.tif") if mask_filter not in f.name
    )

    n_written = 0
    for img_path in image_files:
        mask_path = src_dir / f"{img_path.stem}{mask_filter}.tif"
        vol = tifffile.imread(img_path)     # (Z, Y, X)
        mask = tifffile.imread(mask_path)   # (Z, Y, X)

        for z in range(vol.shape[0]):
            m = mask[z]
            if (len(np.unique(m)) - 1) < min_masks:   # skip empty/sparse slices
                continue
            stem = f"{img_path.stem}_z{z:03d}"
            tifffile.imwrite(dst_dir / f"{stem}.tif", vol[z].astype(np.uint16))
            tifffile.imwrite(dst_dir / f"{stem}{mask_filter}.tif", m.astype(np.uint16))
            n_written += 1

    logger.info(f"Sliced {src_dir.name} -> {dst_dir.name}: {n_written} 2D pairs written")
