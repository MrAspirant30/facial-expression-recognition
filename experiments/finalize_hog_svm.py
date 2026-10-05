"""
Final HOG + Linear SVM Pipeline on FER2013 Clean Split.

TEST SET IS LOCKED FOR MODEL SELECTION. This evaluation is performed only after
hyperparameter selection is complete.

METHODOLOGY & PROTOCOL:
- Hyperparameter selection is finalized (Champion: 900-D HOG, LinearSVC C=0.35, class_weight='balanced').
- Training set: Combines Train (22,347) + Validation (5,586) splits = 27,933 samples.
- Evaluation: Evaluates ONCE on the untouched, locked Test set (7,178 samples).
- Features: Reuses pre-extracted 900-D float32 features from results/features/.
- Output: Saves final model, test metrics, per-class classification report,
  confusion matrix figure, and reproducibility record.
"""

import csv
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import HOG_CONFIG
from src.models.hog_svm import CLASS_NAMES
from src.evaluation.evaluate import plot_and_save_confusion_matrix


# Directory Paths
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
MODELS_DIR = RESULTS_DIR / "models"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Artifact Destinations
MODEL_SAVE_PATH = MODELS_DIR / "hog_svm_final_c035_balanced.joblib"
METRICS_CSV_PATH = TABLES_DIR / "hog_svm_final_test_metrics.csv"
METRICS_JSON_PATH = TABLES_DIR / "hog_svm_final_test_metrics.json"
REPORT_CSV_PATH = TABLES_DIR / "hog_svm_final_classification_report.csv"
REPORT_JSON_PATH = TABLES_DIR / "hog_svm_final_classification_report.json"
FIGURE_PNG_PATH = FIGURES_DIR / "hog_svm_final_confusion_matrix.png"
REPRODUCIBILITY_JSON_PATH = RESULTS_DIR / "reproducibility_hog_svm_final.json"

# Fixed Selected Hyperparameters
SELECTED_HOG_CONFIG: Dict[str, Any] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}
EXPECTED_DIM = 900
EXPECTED_TRAIN_COUNT = 22347
EXPECTED_VAL_COUNT = 5586
EXPECTED_COMBINED_COUNT = 27933
EXPECTED_TEST_COUNT = 7178

SELECTED_SVM_PARAMS: Dict[str, Any] = {
    "classifier": "LinearSVC",
    "C": 0.35,
    "class_weight": "balanced",
    "random_state": 42,
    "max_iter": 2000,
    "dual": "auto",
}

VALIDATION_CHAMPION_METRICS: Dict[str, float] = {
    "accuracy": 0.4076,
    "macro_precision": 0.3463,
    "macro_recall": 0.3930,
    "macro_f1": 0.3463,
}


def load_and_validate_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate train, validation, and test HOG feature matrices and labels.
    """
    train_x_path = FEATURES_DIR / "train_features.npy"
    train_y_path = FEATURES_DIR / "train_labels.npy"
    val_x_path = FEATURES_DIR / "val_features.npy"
    val_y_path = FEATURES_DIR / "val_labels.npy"
    test_x_path = FEATURES_DIR / "test_features.npy"
    test_y_path = FEATURES_DIR / "test_labels.npy"

    for p in (train_x_path, train_y_path, val_x_path, val_y_path, test_x_path, test_y_path):
        if not p.exists():
            raise FileNotFoundError(f"Required cached feature file missing: {p.as_posix()}")

    print("Loading cached HOG feature matrices...")
    t0 = time.time()
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    load_time = time.time() - t0

    print(f"Features loaded in {load_time:.2f}s:")
    print(f"  Train:      X={X_train.shape}, y={y_train.shape}")
    print(f"  Validation: X={X_val.shape},   y={y_val.shape}")
    print(f"  Test:       X={X_test.shape},  y={y_test.shape}")

    # Shape and count validations
    assert X_train.shape == (EXPECTED_TRAIN_COUNT, EXPECTED_DIM), f"Train shape mismatch: {X_train.shape}"
    assert len(y_train) == EXPECTED_TRAIN_COUNT, f"Train label count mismatch: {len(y_train)}"
    assert X_val.shape == (EXPECTED_VAL_COUNT, EXPECTED_DIM), f"Val shape mismatch: {X_val.shape}"
    assert len(y_val) == EXPECTED_VAL_COUNT, f"Val label count mismatch: {len(y_val)}"
    assert X_test.shape == (EXPECTED_TEST_COUNT, EXPECTED_DIM), f"Test shape mismatch: {X_test.shape}"
    assert len(y_test) == EXPECTED_TEST_COUNT, f"Test label count mismatch: {len(y_test)}"

    return X_train, y_train, X_val, y_val, X_test, y_test


def combine_train_and_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine Train and Validation splits into final full training set.
    """
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    assert X_combined.shape == (EXPECTED_COMBINED_COUNT, EXPECTED_DIM), (
        f"Combined shape mismatch: {X_combined.shape}, expected ({EXPECTED_COMBINED_COUNT}, {EXPECTED_DIM})"
    )
    assert len(y_combined) == EXPECTED_COMBINED_COUNT, (
        f"Combined label count mismatch: {len(y_combined)}"
    )

    print(f"\nCombined Train + Validation into final training set: {X_combined.shape} ({X_combined.nbytes / (1024*1024):.1f} MB)")
    return X_combined, y_combined


def train_final_model(X_train_full: np.ndarray, y_train_full: np.ndarray) -> Tuple[LinearSVC, float]:
    """
    Train LinearSVC on combined train+val data with the selected champion hyperparameters.
    """
    print(f"\nTraining final LinearSVC(C=0.35, class_weight='balanced', random_state=42, max_iter=2000, dual='auto')...")
    model = LinearSVC(
        C=SELECTED_SVM_PARAMS["C"],
        class_weight=SELECTED_SVM_PARAMS["class_weight"],
        random_state=SELECTED_SVM_PARAMS["random_state"],
        max_iter=SELECTED_SVM_PARAMS["max_iter"],
        dual=SELECTED_SVM_PARAMS["dual"],
    )

    t_start = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - t_start

    print(f"Final model training complete in {train_time:.2f}s.")
    return model, train_time


def evaluate_final_model(
    model: LinearSVC,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any], float]:
    """
    Evaluate final trained model once on the locked test set.
    Computes overall summary metrics, confusion matrix, and per-class classification report.
    """
    print("\nEvaluating final model on locked test set...")
    t_start = time.time()
    y_pred_test = model.predict(X_test)
    eval_time = time.time() - t_start

    acc = float(accuracy_score(y_test, y_pred_test))
    macro_p = float(precision_score(y_test, y_pred_test, average="macro", zero_division=0))
    macro_r = float(recall_score(y_test, y_pred_test, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred_test, average="macro", zero_division=0))
    cm = confusion_matrix(y_test, y_pred_test)

    report_dict = classification_report(
        y_test,
        y_pred_test,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": round(acc, 6),
        "macro_precision": round(macro_p, 6),
        "macro_recall": round(macro_r, 6),
        "macro_f1": round(macro_f1, 6),
    }

    print(f"Evaluation complete in {eval_time:.3f}s.")
    return metrics, cm, report_dict, eval_time


def save_artifacts(
    model: LinearSVC,
    metrics: Dict[str, Any],
    cm: np.ndarray,
    report_dict: Dict[str, Any],
    train_runtime: float,
    eval_runtime: float,
) -> None:
    """
    Persist all final artifacts: model joblib, metrics tables, classification report,
    confusion matrix figure, and reproducibility metadata.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Model Checkpoint
    model_payload = {
        "classifier": model,
        "hog_config": SELECTED_HOG_CONFIG,
        "class_names": CLASS_NAMES,
        "C": SELECTED_SVM_PARAMS["C"],
        "class_weight": SELECTED_SVM_PARAMS["class_weight"],
        "random_state": SELECTED_SVM_PARAMS["random_state"],
        "max_iter": SELECTED_SVM_PARAMS["max_iter"],
        "dual": SELECTED_SVM_PARAMS["dual"],
    }
    joblib.dump(model_payload, MODEL_SAVE_PATH)
    print(f"[SAVED] Final model checkpoint: {MODEL_SAVE_PATH.as_posix()}")

    # 2. Save Final Test Metrics Table (CSV & JSON)
    metrics_record = {
        "model_name": "Final HOG + Linear SVM (C=0.35, balanced)",
        "dataset": "FER2013 (clean_dataset_split)",
        "hog_orientations": SELECTED_HOG_CONFIG["orientations"],
        "hog_pixels_per_cell": str(SELECTED_HOG_CONFIG["pixels_per_cell"]),
        "hog_cells_per_block": str(SELECTED_HOG_CONFIG["cells_per_block"]),
        "hog_block_norm": SELECTED_HOG_CONFIG["block_norm"],
        "hog_feature_dimension": EXPECTED_DIM,
        "classifier": "LinearSVC",
        "svm_C": SELECTED_SVM_PARAMS["C"],
        "svm_class_weight": SELECTED_SVM_PARAMS["class_weight"],
        "svm_random_state": SELECTED_SVM_PARAMS["random_state"],
        "svm_max_iter": SELECTED_SVM_PARAMS["max_iter"],
        "svm_dual": SELECTED_SVM_PARAMS["dual"],
        "train_samples": EXPECTED_TRAIN_COUNT,
        "validation_samples": EXPECTED_VAL_COUNT,
        "combined_training_samples": EXPECTED_COMBINED_COUNT,
        "test_samples": EXPECTED_TEST_COUNT,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "training_runtime_seconds": round(train_runtime, 2),
        "evaluation_runtime_seconds": round(eval_runtime, 3),
    }

    with open(METRICS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_record.keys()))
        writer.writeheader()
        writer.writerow(metrics_record)

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    print(f"[SAVED] Final test metrics: {METRICS_CSV_PATH.as_posix()} and .json")

    # 3. Save Classification Report (CSV & JSON)
    report_rows = []
    for cls_name in CLASS_NAMES:
        cls_data = report_dict[cls_name]
        report_rows.append({
            "class_name": cls_name,
            "precision": round(cls_data["precision"], 4),
            "recall": round(cls_data["recall"], 4),
            "f1_score": round(cls_data["f1-score"], 4),
            "support": int(cls_data["support"]),
        })

    # Add Macro Avg and Weighted Avg
    for avg_key, avg_label in [("macro avg", "macro_avg"), ("weighted avg", "weighted_avg")]:
        avg_data = report_dict[avg_key]
        report_rows.append({
            "class_name": avg_label,
            "precision": round(avg_data["precision"], 4),
            "recall": round(avg_data["recall"], 4),
            "f1_score": round(avg_data["f1-score"], 4),
            "support": int(avg_data["support"]),
        })

    with open(REPORT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "precision", "recall", "f1_score", "support"])
        writer.writeheader()
        writer.writerows(report_rows)

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"[SAVED] Classification report: {REPORT_CSV_PATH.as_posix()} and .json")

    # 4. Save Confusion Matrix Figure
    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=str(FIGURE_PNG_PATH),
        title="FER2013 Final Test Confusion Matrix - HOG + Linear SVM (C=0.35, balanced)",
    )
    print(f"[SAVED] Confusion matrix figure: {FIGURE_PNG_PATH.as_posix()}")

    # 5. Save Reproducibility Metadata
    reproducibility_payload = {
        "model_name": "Final HOG + Linear SVM (C=0.35, balanced)",
        "methodology": "TEST SET IS LOCKED FOR MODEL SELECTION. This evaluation is performed only after hyperparameter selection is complete.",
        "hog_configuration": {
            "orientations": SELECTED_HOG_CONFIG["orientations"],
            "pixels_per_cell": list(SELECTED_HOG_CONFIG["pixels_per_cell"]),
            "cells_per_block": list(SELECTED_HOG_CONFIG["cells_per_block"]),
            "block_norm": SELECTED_HOG_CONFIG["block_norm"],
        },
        "svm_configuration": SELECTED_SVM_PARAMS,
        "training_samples": EXPECTED_TRAIN_COUNT,
        "validation_samples": EXPECTED_VAL_COUNT,
        "combined_training_samples": EXPECTED_COMBINED_COUNT,
        "test_samples": EXPECTED_TEST_COUNT,
        "feature_dimension": EXPECTED_DIM,
        "random_state": SELECTED_SVM_PARAMS["random_state"],
        "selection_metric": "Validation Macro F1",
        "selected_validation_metrics": VALIDATION_CHAMPION_METRICS,
        "final_test_metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }

    with open(REPRODUCIBILITY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(reproducibility_payload, f, indent=2)

    print(f"[SAVED] Reproducibility record: {REPRODUCIBILITY_JSON_PATH.as_posix()}")


def main():
    print("=" * 65)
    print("FINAL HOG + LINEAR SVM EXECUTION PROTOCOL")
    print("=" * 65)
    print("TEST SET IS LOCKED FOR MODEL SELECTION. This evaluation is performed only after hyperparameter selection is complete.\n")

    # 1. Load cached features
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_validate_features()

    # 2. Combine Train + Validation splits
    X_train_full, y_train_full = combine_train_and_val(X_train, y_train, X_val, y_val)

    # 3. Train final model on combined data
    model, train_time = train_final_model(X_train_full, y_train_full)

    # 4. Evaluate once on locked test set
    metrics, cm, report_dict, eval_time = evaluate_final_model(model, X_test, y_test)

    # 5. Persist all final artifacts
    save_artifacts(model, metrics, cm, report_dict, train_time, eval_time)

    # 6. Terminal Summary Printout
    print("\n" + "=" * 50)
    print("FINAL HOG + LINEAR SVM")
    print("=" * 50)
    print("HOG: 9 orientations, 8x8 cell, 2x2 block, L2-Hys")
    print("SVM: LinearSVC")
    print("C: 0.35")
    print("class_weight: balanced")
    print(f"Training samples: {EXPECTED_COMBINED_COUNT}")
    print(f"Test samples: {EXPECTED_TEST_COUNT}")
    print(f"Feature dimension: {EXPECTED_DIM}\n")
    print("FINAL TEST RESULTS")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
