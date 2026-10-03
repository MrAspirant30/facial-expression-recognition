"""
HOG + Linear SVM Model Module.

Provides training, inference, and persistence for a Linear Support Vector Machine
operating on HOG feature vectors.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.svm import LinearSVC

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import HOG_CONFIG, extract_hog_features
from src.evaluation.evaluate import evaluate_predictions


def extract_features_batch(
    images: Union[np.ndarray, List[np.ndarray]],
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Extract 1D HOG feature vectors for a collection of images and stack into a 2D matrix.

    Args:
        images: Array or sequence of standardized 2D or 3D images.
        config: Optional HOG configuration dictionary.

    Returns:
        2D numpy array of shape (N, feature_dim).
    """
    if config is None:
        config = HOG_CONFIG

    feature_list = [extract_hog_features(img, config=config) for img in images]
    return np.vstack(feature_list)


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


class HOGLinearSVM:
    """Linear Support Vector Classifier pipeline using HOG feature representations."""

    def __init__(
        self,
        C: float = 1.0,
        random_state: int = 42,
        max_iter: int = 2000,
        hog_config: Optional[Dict[str, Any]] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.C = C
        self.random_state = random_state
        self.max_iter = max_iter
        self.hog_config = dict(hog_config) if hog_config is not None else dict(HOG_CONFIG)
        self.class_names = list(class_names) if class_names is not None else list(CLASS_NAMES)
        self.classifier = LinearSVC(
            C=self.C,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

    def extract_features(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Extract HOG features from images using this model's configuration."""
        return extract_features_batch(images, config=self.hog_config)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HOGLinearSVM":
        """Fit LinearSVC on pre-extracted feature matrix X and target labels y."""
        self.classifier.fit(X, y)
        return self

    def fit_images(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        y: np.ndarray,
    ) -> Tuple["HOGLinearSVM", np.ndarray]:
        """Extract HOG features from raw images and fit the LinearSVC."""
        X = self.extract_features(images)
        self.fit(X, y)
        return self, X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for pre-extracted feature matrix X."""
        return self.classifier.predict(X)

    def predict_images(self, images: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Extract HOG features from images and predict class labels."""
        X = self.extract_features(images)
        return self.predict(X)

    def predict_single_image(self, image: np.ndarray) -> Tuple[int, str]:
        """
        Predict facial expression for a single image.

        Args:
            image: Standardized 2D image (H, W) or single-channel 3D image.

        Returns:
            Tuple of (predicted_class_id, predicted_class_name).
        """
        feat = extract_hog_features(image, config=self.hog_config)
        pred_label = int(self.classifier.predict(feat.reshape(1, -1))[0])
        class_name = (
            self.class_names[pred_label]
            if 0 <= pred_label < len(self.class_names)
            else str(pred_label)
        )
        return pred_label, class_name

    def save(self, filepath: Union[str, Path]) -> None:
        """Save trained classifier along with its associated HOG configuration."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifier": self.classifier,
            "hog_config": self.hog_config,
            "class_names": self.class_names,
            "C": self.C,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
        }
        joblib.dump(payload, save_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HOGLinearSVM":
        """Load trained classifier and its associated HOG configuration."""
        payload = joblib.load(filepath)
        instance = cls(
            C=payload.get("C", 1.0),
            random_state=payload.get("random_state", 42),
            max_iter=payload.get("max_iter", 2000),
            hog_config=payload.get("hog_config", HOG_CONFIG),
            class_names=payload.get("class_names", CLASS_NAMES),
        )
        instance.classifier = payload["classifier"]
        return instance


def run_smoke_test() -> None:
    """Run end-to-end smoke test on a synthetic multi-class image set."""
    rng = np.random.RandomState(42)

    # Generate synthetic 48x48 images across 3 distinct visual pattern classes
    samples_per_class = 6
    num_classes = 3

    images = []
    labels = []

    for c in range(num_classes):
        for _ in range(samples_per_class):
            img = rng.normal(loc=0.5, scale=0.05, size=(48, 48)).astype(np.float32)
            if c == 0:
                # Class 0: Horizontal stripe patterns
                img[::4, :] += 0.4
            elif c == 1:
                # Class 1: Vertical stripe patterns
                img[:, ::4] += 0.4
            else:
                # Class 2: Concentric center box pattern
                img[16:32, 16:32] += 0.4
            img = np.clip(img, 0.0, 1.0)
            images.append(img)
            labels.append(c)

    images_arr = np.array(images)
    y_true = np.array(labels)

    # 1. 48x48 sample image -> existing extract_hog_features() -> HOG feature vector
    single_sample_feat = extract_hog_features(images_arr[0])

    # 2. Extract feature matrix for the image set using extract_hog_features
    model = HOGLinearSVM(C=1.0, random_state=42)
    X = model.extract_features(images_arr)

    # 3. Train HOGLinearSVM
    model.fit(X, y_true)
    training_completed = hasattr(model.classifier, "coef_")

    # 4. Predict
    y_pred = model.predict(X)

    # 5. Evaluate
    metrics = evaluate_predictions(y_true, y_pred)

    print("=== SMOKE TEST ONLY -- NOT FER2013 RESULTS ===")
    print(f"Image array shape: {images_arr.shape}")
    print(f"HOG feature shape: {single_sample_feat.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Training completed: {training_completed}")
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Confusion matrix shape: {metrics['confusion_matrix'].shape}")


if __name__ == "__main__":
    run_smoke_test()
