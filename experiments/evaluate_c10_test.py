"""
Final Test Evaluation for Selected Hyperparameter C = 10.0.

Loads cached training features, trains LinearSVC(C=10.0, random_state=42),
evaluates on the untouched FER2013 test set (7,178 samples), and exports
new test metrics, confusion matrix figure, and saved model checkpoint.
"""

import csv
import gc
import json
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import HOG_CONFIG, extract_hog_features
from src.models.hog_svm import CLASS_NAMES, HOGLinearSVM
from src.evaluation.evaluate import evaluate_predictions, plot_and_save_confusion_matrix


MANIFEST_PATH = PROJECT_ROOT / "data" / "fer2013" / "archive" / "clean_dataset_split.csv"
DATASET_BASE_DIR = MANIFEST_PATH.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
FEATURES_DIR = RESULTS_DIR / "features"


def load_test_rows(manifest_path: Path) -> List[dict]:
    """Read manifest and collect only test split rows."""
    test_rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test":
                test_rows.append(row)
    return test_rows


def extract_or_load_test_features(
    test_rows: List[dict],
    base_dir: Path,
    expected_dim: int = 900,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached test features if available, else extract and cache as .npy."""
    x_cache_path = FEATURES_DIR / "test_features.npy"
    y_cache_path = FEATURES_DIR / "test_labels.npy"

    if x_cache_path.exists() and y_cache_path.exists():
        print(f"Loading cached test features from {x_cache_path}...")
        X_test = np.load(x_cache_path)
        y_test = np.load(y_cache_path)
        if len(X_test) == len(test_rows) and X_test.shape[1] == expected_dim:
            print(f"Cache hit for test: {X_test.shape}, dtype={X_test.dtype}")
            return X_test, y_test
        print("Test cache mismatch, re-extracting...")

    n_samples = len(test_rows)
    X_test = np.empty((n_samples, expected_dim), dtype=np.float32)
    y_test = np.empty(n_samples, dtype=np.int64)

    start_time = time.time()
    for idx, row in enumerate(test_rows):
        img_path = base_dir / row["path"]
        with Image.open(img_path) as img:
            arr = np.asarray(img, dtype=np.float32) / 255.0

        feat = extract_hog_features(arr, config=HOG_CONFIG)
        X_test[idx] = feat
        y_test[idx] = int(row["label"])

        if (idx + 1) % 2500 == 0 or (idx + 1) == n_samples:
            elapsed = time.time() - start_time
            print(f"[TEST] Extracted {idx + 1}/{n_samples} samples ({elapsed:.1f}s)...")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(x_cache_path, X_test)
    np.save(y_cache_path, y_test)
    print(f"Cached test features to {x_cache_path} ({X_test.nbytes / (1024*1024):.2f} MB)")

    gc.collect()
    return X_test, y_test


def main():
    print("=" * 65)
    print("FINAL FER2013 TEST EVALUATION: HOG + LinearSVC (C = 10.0)")
    print("=" * 65)

    # 1. Load cached training features
    train_x_cache = FEATURES_DIR / "train_features.npy"
    train_y_cache = FEATURES_DIR / "train_labels.npy"
    assert train_x_cache.exists() and train_y_cache.exists(), "Cached train features missing!"

    print(f"\n1. Loading cached training features from {train_x_cache}...")
    X_train = np.load(train_x_cache)
    y_train = np.load(train_y_cache)
    print(f"   X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"   y_train shape: {y_train.shape}")
    assert len(X_train) == 22347, f"Expected 22347 train samples, got {len(X_train)}"

    # 2. Load and extract test set features
    print(f"\n2. Loading test set (7,178 samples)...")
    test_rows = load_test_rows(MANIFEST_PATH)
    assert len(test_rows) == 7178, f"Expected 7178 test samples, got {len(test_rows)}"

    X_test, y_test = extract_or_load_test_features(test_rows, DATASET_BASE_DIR, expected_dim=900)
    print(f"   X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"   y_test shape: {y_test.shape}")
    assert len(X_test) == 7178, f"Expected 7178 test samples, got {len(X_test)}"

    # 3. Train LinearSVC with C = 10.0
    print(f"\n3. Training LinearSVC on X_train (C=10.0, random_state=42, max_iter=2000)...")
    svm_model = HOGLinearSVM(
        C=10.0,
        random_state=42,
        max_iter=2000,
        hog_config=HOG_CONFIG,
        class_names=CLASS_NAMES,
    )

    t_start = time.time()
    svm_model.fit(X_train, y_train)
    training_runtime = time.time() - t_start
    print(f"   LinearSVC(C=10.0) training completed in {training_runtime:.2f}s.")

    # 4. Predict on untouched Test set
    print(f"\n4. Evaluating on untouched test set ({len(X_test)} samples)...")
    t_eval = time.time()
    y_pred_test = svm_model.predict(X_test)
    eval_runtime = time.time() - t_eval

    # 5. Calculate Metrics
    metrics = evaluate_predictions(y_test, y_pred_test)
    accuracy = metrics["accuracy"]
    macro_precision = metrics["macro_precision"]
    macro_recall = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 65)
    print("FINAL C=10 FER2013 TEST RESULTS")
    print("=" * 65)
    print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Confusion Matrix Shape: {cm.shape}")
    print("Confusion Matrix:")
    print(cm)

    # 6. Sanity Checks
    print(f"\n5. Running Sanity Checks...")
    assert len(y_train) == 22347, "Training samples count mismatch!"
    assert len(y_test) == 7178, "Test samples count mismatch!"
    assert X_train.shape[1] == 900, "X_train feature dimension mismatch!"
    assert X_test.shape[1] == 900, "X_test feature dimension mismatch!"
    assert len(y_pred_test) == 7178, "Prediction count mismatch!"
    assert cm.shape == (7, 7), "Confusion matrix must be 7x7!"
    print("   All sanity checks PASSED.")

    # 7. Save Model
    model_save_path = MODELS_DIR / "hog_svm_model_c10.joblib"
    svm_model.save(model_save_path)
    print(f"\n6. Saved model to: {model_save_path}")

    # 8. Save Metrics Table (CSV & JSON)
    table_csv_path = TABLES_DIR / "hog_svm_c10_test_metrics.csv"
    table_json_path = TABLES_DIR / "hog_svm_c10_test_metrics.json"

    result_record = {
        "model_name": "HOG + Linear SVM (C=10.0)",
        "dataset": "FER2013 (clean_dataset_split)",
        "train_samples": len(X_train),
        "validation_samples": 5586,
        "test_samples": len(X_test),
        "hog_orientations": HOG_CONFIG["orientations"],
        "hog_pixels_per_cell": str(HOG_CONFIG["pixels_per_cell"]),
        "hog_cells_per_block": str(HOG_CONFIG["cells_per_block"]),
        "hog_block_norm": HOG_CONFIG.get("block_norm", "L2-Hys"),
        "hog_feature_dimension": 900,
        "svm_classifier": "LinearSVC",
        "svm_C": 10.0,
        "svm_random_state": 42,
        "svm_max_iter": 2000,
        "accuracy": round(accuracy, 6),
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "training_runtime_seconds": round(training_runtime, 2),
    }

    with open(table_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result_record.keys()))
        writer.writeheader()
        writer.writerow(result_record)
    print(f"   Saved test metrics table (CSV) to: {table_csv_path}")

    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)
    print(f"   Saved test metrics table (JSON) to: {table_json_path}")

    # 9. Save Confusion Matrix Figure
    cm_figure_path = FIGURES_DIR / "confusion_matrix_c10.png"
    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=str(cm_figure_path),
        title="FER2013 Test Confusion Matrix - HOG + Linear SVM (C=10.0)",
    )
    print(f"   Saved confusion matrix figure to: {cm_figure_path}")

    # 10. Baseline Comparison
    baseline_csv_path = TABLES_DIR / "hog_svm_test_metrics.csv"
    if baseline_csv_path.exists():
        with open(baseline_csv_path, "r", encoding="utf-8") as f:
            b_reader = csv.DictReader(f)
            b_row = next(b_reader)
            b_acc = float(b_row["accuracy"])
            b_mp = float(b_row["macro_precision"])
            b_mr = float(b_row["macro_recall"])
            b_mf1 = float(b_row["macro_f1"])

        print("\n" + "=" * 65)
        print("COMPARISON: BASELINE (C=1.0) vs TUNED (C=10.0) ON TEST SET")
        print("=" * 65)
        print(f"{'Metric':<18} | {'Baseline (C=1.0)':<18} | {'Tuned (C=10.0)':<18} | {'Delta':<10}")
        print("-" * 65)
        print(f"{'Accuracy':<18} | {b_acc:<18.4f} | {accuracy:<18.4f} | {accuracy - b_acc:+10.4f}")
        print(f"{'Macro Precision':<18} | {b_mp:<18.4f} | {macro_precision:<18.4f} | {macro_precision - b_mp:+10.4f}")
        print(f"{'Macro Recall':<18} | {b_mr:<18.4f} | {macro_recall:<18.4f} | {macro_recall - b_mr:+10.4f}")
        print(f"{'Macro F1':<18} | {b_mf1:<18.4f} | {macro_f1:<18.4f} | {macro_f1 - b_mf1:+10.4f}")
        print("-" * 65)


if __name__ == "__main__":
    main()
