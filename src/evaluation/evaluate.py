"""
Evaluation Module.

Provides standard evaluation metrics and confusion matrix computation
using scikit-learn for facial expression classification.
"""

from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate classification accuracy."""
    return float(accuracy_score(y_true, y_pred))


def compute_macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate macro-averaged precision across all classes."""
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def compute_macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate macro-averaged recall across all classes."""
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate macro-averaged F1 score across all classes."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix from true and predicted labels."""
    return confusion_matrix(y_true, y_pred)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Compute all required evaluation metrics and return as a dictionary.

    Args:
        y_true: Ground truth target labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, macro precision, macro recall,
        macro F1, and confusion matrix.
    """
    return {
        "accuracy": compute_accuracy(y_true, y_pred),
        "macro_precision": compute_macro_precision(y_true, y_pred),
        "macro_recall": compute_macro_recall(y_true, y_pred),
        "macro_f1": compute_macro_f1(y_true, y_pred),
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred),
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str,
    title: str = "Confusion Matrix - HOG + Linear SVM",
) -> None:
    """
    Plot and save confusion matrix as an image file.

    Args:
        cm: Confusion matrix array of shape (num_classes, num_classes).
        class_names: List of class label strings.
        output_path: Path where figure image will be saved.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    num_classes = len(class_names)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells with integer counts
    thresh = cm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
