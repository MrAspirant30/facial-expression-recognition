"""
Full Pipeline Runner for HOG + Linear SVM on Clean FER2013.

Loads data using the authoritative manifest, performs memory-conscious
HOG feature extraction, trains LinearSVC on X_train/y_train only,
evaluates on the untouched test set, generates results tables and confusion
matrix figures, and tests single-image prediction.
"""

import csv
import gc
import json
from pathlib import Path
import platform
import sys
import time
from typing import Dict, List, Tuple

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


def load_split_rows(manifest_path: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    """Read manifest and partition rows into train, validation, and test splits."""
    train_rows, val_rows, test_rows = [], [], []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split == "train":
                train_rows.append(row)
            elif split == "validation":
                val_rows.append(row)
            elif split == "test":
                test_rows.append(row)

    return train_rows, val_rows, test_rows


def extract_features_from_rows(
    rows: List[dict],
    base_dir: Path,
    expected_dim: int = 900,
    split_name: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-conscious HOG extraction directly into a pre-allocated float32 array.
    Avoids retaining duplicate raw images in memory.
    """
    n_samples = len(rows)
    X = np.empty((n_samples, expected_dim), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)

    start_time = time.time()
    for idx, row in enumerate(rows):
        img_path = base_dir / row["path"]
        with Image.open(img_path) as img:
            arr = np.asarray(img, dtype=np.float32) / 255.0

        feat = extract_hog_features(arr, config=HOG_CONFIG)
        X[idx] = feat
        y[idx] = int(row["label"])

        if (idx + 1) % 5000 == 0 or (idx + 1) == n_samples:
            elapsed = time.time() - start_time
            print(f"[{split_name.upper()}] Processed {idx + 1}/{n_samples} samples ({elapsed:.1f}s)...")

    gc.collect()
    return X, y


def main():
    print("=" * 65)
    print("FER2013 HOG + LINEAR SVM - FULL TRAINING & EVALUATION RUN")
    print("=" * 65)

    # 1. Verify Manifest and Split Counts
    print(f"\n1. Verifying dataset manifest: {MANIFEST_PATH.as_posix()}")
    assert MANIFEST_PATH.exists(), f"Manifest file not found: {MANIFEST_PATH}"

    train_rows, val_rows, test_rows = load_split_rows(MANIFEST_PATH)
    print(f"Train samples:      {len(train_rows)} (Expected: 22347)")
    print(f"Validation samples: {len(val_rows)}  (Expected: 5586)")
    print(f"Test samples:       {len(test_rows)}  (Expected: 7178)")

    if len(train_rows) != 22347 or len(val_rows) != 5586 or len(test_rows) != 7178:
        raise ValueError("Manifest split counts do not match expected authoritative counts!")

    # Verify sample image feature dimension dynamically from actual extractor
    sample_probe_img = np.zeros((48, 48), dtype=np.float32)
    sample_feat = extract_hog_features(sample_probe_img, config=HOG_CONFIG)
    feature_dim = sample_feat.shape[0]
    print(f"\n2. HOG Feature Extractor:")
    print(f"   Configuration:     {HOG_CONFIG}")
    print(f"   Measured 1D dim:   {feature_dim}")

    # 2. Memory-Conscious Feature Extraction for Train and Test
    print(f"\n3. Extracting HOG features for Train set ({len(train_rows)} samples)...")
    extract_start = time.time()
    X_train, y_train = extract_features_from_rows(
        train_rows,
        DATASET_BASE_DIR,
        expected_dim=feature_dim,
        split_name="train",
    )
    extract_train_time = time.time() - extract_start
    print(f"   X_train shape: {X_train.shape}, dtype: {X_train.dtype}, size: {X_train.nbytes / (1024*1024):.2f} MB")
    print(f"   Extraction time: {extract_train_time:.2f}s")

    print(f"\n4. Extracting HOG features for Test set ({len(test_rows)} samples)...")
    test_extract_start = time.time()
    X_test, y_test = extract_features_from_rows(
        test_rows,
        DATASET_BASE_DIR,
        expected_dim=feature_dim,
        split_name="test",
    )
    extract_test_time = time.time() - test_extract_start
    print(f"   X_test shape: {X_test.shape}, dtype: {X_test.dtype}, size: {X_test.nbytes / (1024*1024):.2f} MB")
    print(f"   Extraction time: {extract_test_time:.2f}s")

    # 3. Train LinearSVC on X_train, y_train only
    print(f"\n5. Training LinearSVC on X_train (C=1.0, random_state=42)...")
    svm_model = HOGLinearSVM(C=1.0, random_state=42, max_iter=2000, hog_config=HOG_CONFIG, class_names=CLASS_NAMES)

    train_start = time.time()
    svm_model.fit(X_train, y_train)
    training_runtime = time.time() - train_start
    print(f"   LinearSVC training completed in {training_runtime:.2f}s.")

    # 4. Final Test Evaluation
    print(f"\n6. Evaluating on untouched official Test set ({len(y_test)} samples)...")
    eval_start = time.time()
    y_pred_test = svm_model.predict(X_test)
    eval_runtime = time.time() - eval_start

    metrics = evaluate_predictions(y_test, y_pred_test)
    accuracy = metrics["accuracy"]
    macro_precision = metrics["macro_precision"]
    macro_recall = metrics["macro_recall"]
    macro_f1 = metrics["macro_f1"]
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 65)
    print("ACTUAL FER2013 TEST RESULTS")
    print("=" * 65)
    print(f"Accuracy:         {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Confusion Matrix Shape: {cm.shape}")
    print("Confusion Matrix:")
    print(cm)

    # 5. Sanity Checks
    print(f"\n7. Running Final Sanity Checks...")
    assert len(y_train) == 22347, "Train sample count mismatch!"
    assert len(y_test) == 7178, "Test sample count mismatch!"
    assert X_train.shape[1] == feature_dim, "X_train feature dimension mismatch!"
    assert X_test.shape[1] == feature_dim, "X_test feature dimension mismatch!"
    assert len(y_pred_test) == len(y_test), "Prediction length mismatch!"
    assert cm.shape == (7, 7), "Confusion matrix must be 7x7!"
    assert set(np.unique(y_test)).issubset(range(7)), "Invalid labels in test set!"
    print("   All sanity checks PASSED.")

    # 6. Save Model
    model_save_path = MODELS_DIR / "hog_svm_model.joblib"
    print(f"\n8. Saving trained model to {model_save_path}...")
    svm_model.save(model_save_path)

    # Verify single-image prediction interface with saved model
    loaded_model = HOGLinearSVM.load(model_save_path)
    sample_test_img_path = DATASET_BASE_DIR / test_rows[0]["path"]
    with Image.open(sample_test_img_path) as s_img:
        sample_arr = np.asarray(s_img, dtype=np.float32) / 255.0
    pred_idx, pred_name = loaded_model.predict_single_image(sample_arr)
    print(f"   Single-image prediction verified:")
    print(f"   Sample path: {test_rows[0]['path']}")
    print(f"   Ground truth: {test_rows[0]['label']} ({test_rows[0]['class']})")
    print(f"   Predicted:    {pred_idx} ({pred_name})")

    # 7. Save Results Table
    table_csv_path = TABLES_DIR / "hog_svm_test_metrics.csv"
    table_json_path = TABLES_DIR / "hog_svm_test_metrics.json"

    result_record = {
        "model_name": "HOG + Linear SVM",
        "dataset": "FER2013 (clean_dataset_split)",
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "test_samples": len(test_rows),
        "hog_orientations": HOG_CONFIG["orientations"],
        "hog_pixels_per_cell": str(HOG_CONFIG["pixels_per_cell"]),
        "hog_cells_per_block": str(HOG_CONFIG["cells_per_block"]),
        "hog_block_norm": HOG_CONFIG.get("block_norm", "L2-Hys"),
        "hog_feature_dimension": feature_dim,
        "svm_classifier": "LinearSVC",
        "svm_C": svm_model.C,
        "svm_random_state": svm_model.random_state,
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
    print(f"\n9. Results table saved to: {table_csv_path}")

    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(result_record, f, indent=2)

    # 8. Save Confusion Matrix Figure
    cm_figure_path = FIGURES_DIR / "confusion_matrix.png"
    plot_and_save_confusion_matrix(
        cm=cm,
        class_names=CLASS_NAMES,
        output_path=str(cm_figure_path),
        title="FER2013 Test Confusion Matrix - HOG + Linear SVM",
    )
    print(f"   Confusion matrix figure saved to: {cm_figure_path}")

    # 9. Save Reproducibility Record
    reproducibility_record = {
        "dataset_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "dataset_counts": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "test": len(test_rows),
        },
        "image_dimensions": [48, 48],
        "image_preprocessing": "scaled to [0.0, 1.0], grayscale float32",
        "hog_parameters": HOG_CONFIG,
        "hog_feature_dimension": feature_dim,
        "svm_parameters": {
            "classifier": "LinearSVC",
            "C": svm_model.C,
            "random_state": svm_model.random_state,
            "max_iter": svm_model.max_iter,
        },
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.system() + " " + platform.release(),
            "processor": platform.processor(),
        },
        "runtimes": {
            "train_hog_extraction_seconds": round(extract_train_time, 2),
            "test_hog_extraction_seconds": round(extract_test_time, 2),
            "svm_training_runtime_seconds": round(training_runtime, 2),
            "evaluation_runtime_seconds": round(eval_runtime, 2),
        },
        "actual_test_metrics": {
            "accuracy": round(accuracy, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
        },
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
    }

    reproducibility_path = RESULTS_DIR / "reproducibility.json"
    with open(reproducibility_path, "w", encoding="utf-8") as f:
        json.dump(reproducibility_record, f, indent=2)
    print(f"   Reproducibility record saved to: {reproducibility_path}")
    print("\nPipeline run complete!")


if __name__ == "__main__":
    main()
