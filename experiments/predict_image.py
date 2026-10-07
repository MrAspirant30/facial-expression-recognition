"""
Single-Image Inference Demo for HOG + Linear SVM.

Loads the trained model checkpoint, processes a single input image to the expected
48x48 grayscale float32 format, extracts HOG features, and outputs the predicted
facial expression.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hog_svm import HOGLinearSVM

DEFAULT_MODEL_PATH = PROJECT_ROOT / "results" / "models" / "hog_svm_model.joblib"


def predict_image(image_path: Path, model_path: Path = DEFAULT_MODEL_PATH):
    """Load model, preprocess image to standard 48x48 float32, and predict expression."""
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")

    # 1. Detect face, crop, resize to 48x48, and normalize to float32
    with Image.open(image_path) as img:
        from src.features.face_detection import preprocess_face_image
        model_asset_path = str(PROJECT_ROOT / "blaze_face_short_range.tflite")
        img_resized = preprocess_face_image(img, model_asset_path=model_asset_path)
        img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0

    # 2. Load model
    model = HOGLinearSVM.load(model_path)

    # 3. Predict expression
    pred_class_id, pred_expression = model.predict_single_image(img_arr)

    # 4. Print results
    print(f"Image: {image_path.as_posix()}")
    print(f"Predicted class ID: {pred_class_id}")
    print(f"Predicted expression: {pred_expression}")

    return pred_class_id, pred_expression


def main():
    parser = argparse.ArgumentParser(description="Predict facial expression for a single image.")
    parser.add_argument("image_path", type=str, help="Path to input image.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model .joblib checkpoint.",
    )
    args = parser.parse_args()

    predict_image(Path(args.image_path), Path(args.model_path))


if __name__ == "__main__":
    main()
