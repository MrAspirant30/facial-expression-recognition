"""
HOG Feature Extraction Module.

Provides configuration and extraction of Histogram of Oriented Gradients (HOG)
features for standardized facial images.
"""

from typing import Any, Dict, Optional
import numpy as np
from skimage.feature import hog


# Central HOG Configuration
HOG_CONFIG: Dict[str, Any] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def extract_hog_features(image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    Extract a 1D HOG feature vector from a single standardized image.

    Args:
        image: Standardized 2D image array (H, W) or 3D single-channel array (H, W, 1).
        config: Optional configuration dictionary. Uses HOG_CONFIG if None.

    Returns:
        1D numpy array of finite numeric HOG features.
    """
    if config is None:
        config = HOG_CONFIG

    img = np.asarray(image)

    # Standardize single-channel 3D image to 2D
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)

    features = hog(
        img,
        orientations=config["orientations"],
        pixels_per_cell=config["pixels_per_cell"],
        cells_per_block=config["cells_per_block"],
        block_norm=config.get("block_norm", "L2-Hys"),
        feature_vector=True,
    )

    # Ensure 1D numpy array with float32 precision
    feat_1d = np.asarray(features, dtype=np.float32).ravel()

    if not np.all(np.isfinite(feat_1d)):
        raise ValueError("Extracted HOG features contain non-finite values (NaN/Inf).")

    return feat_1d


def run_smoke_test() -> None:
    """Run minimal smoke test verifying image -> HOG -> 1D feature vector."""
    # Synthetic 48x48 sample image representing standardized face resolution
    sample_image = np.zeros((48, 48), dtype=np.float32)
    sample_image[10:38, 10:38] = 0.5
    sample_image[20:28, 20:28] = 1.0

    features = extract_hog_features(sample_image, HOG_CONFIG)

    print(f"Input image shape: {sample_image.shape}")
    print(f"Input image dtype: {sample_image.dtype}")
    print(f"HOG parameters: {HOG_CONFIG}")
    print(f"HOG feature shape: {features.shape}")
    print(f"Is feature vector 1D: {features.ndim == 1}")
    print(f"Contains finite values: {bool(np.all(np.isfinite(features)))}")


if __name__ == "__main__":
    run_smoke_test()
