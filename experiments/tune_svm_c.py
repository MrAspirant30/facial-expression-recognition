"""
Controlled Hyperparameter Tuning Experiment: LinearSVC Regularization Parameter C.

Evaluates C in [0.1, 1.0, 10.0] on the clean FER2013 validation set.
Reuses HOG feature extraction once across all candidates.
Does NOT load or evaluate on the test set.
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
from src.models.hog_svm import HOGLinearSVM
from src.evaluation.evaluate import evaluate_predictions


MANIFEST_PATH = PROJECT_ROOT / "data" / "fer2013" / "archive" / "clean_dataset_split.csv"
DATASET_BASE_DIR = MANIFEST_PATH.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FEATURES_DIR = RESULTS_DIR / "features"


def load_train_and_val_rows(manifest_path: Path) -> Tuple[List[dict], List[dict]]:
    """Read manifest and collect only train and validation rows. Strictly ignores test split."""
    train_rows, val_rows = [], []

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split == "train":
                train_rows.append(row)
            elif split == "validation":
                val_rows.append(row)

    return train_rows, val_rows


def extract_or_load_features(
    rows: List[dict],
    cache_prefix: str,
    base_dir: Path,
    expected_dim: int = 900,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached HOG features if available, else extract and cache as .npy."""
    x_cache_path = FEATURES_DIR / f"{cache_prefix}_features.npy"
    y_cache_path = FEATURES_DIR / f"{cache_prefix}_labels.npy"

    if x_cache_path.exists() and y_cache_path.exists():
        print(f"Loading cached {cache_prefix} features from {x_cache_path}...")
        X = np.load(x_cache_path)
        y = np.load(y_cache_path)
        if len(X) == len(rows) and X.shape[1] == expected_dim:
            print(f"Cache hit: {X.shape}, dtype={X.dtype}")
            return X, y
        print("Cache mismatch, re-extracting...")

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
            print(f"[{cache_prefix.upper()}] Extracted {idx + 1}/{n_samples} samples ({elapsed:.1f}s)...")

    # Cache for reuse
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(x_cache_path, X)
    np.save(y_cache_path, y)
    print(f"Cached {cache_prefix} features to {x_cache_path} ({X.nbytes / (1024*1024):.2f} MB)")

    gc.collect()
    return X, y


def run_c_tuning():
    print("=" * 65)
    print("CONTROLLED EXPERIMENT: SVM C HYPERPARAMETER TUNING")
    print("=" * 65)
    print("Strict Constraint: Using Train and Validation sets only. Test set is ISOLATED.\n")

    # 1. Load Train and Validation sets
    train_rows, val_rows = load_train_and_val_rows(MANIFEST_PATH)
    print(f"Train samples:      {len(train_rows)} (Expected: 22347)")
    print(f"Validation samples: {len(val_rows)}  (Expected: 5586)")
    assert len(train_rows) == 22347, "Train count mismatch!"
    assert len(val_rows) == 5586, "Validation count mismatch!"

    # 2. Extract or load HOG features
    feature_dim = 900
    X_train, y_train = extract_or_load_features(train_rows, "train", DATASET_BASE_DIR, feature_dim)
    X_val, y_val = extract_or_load_features(val_rows, "val", DATASET_BASE_DIR, feature_dim)

    # 3. Candidates to evaluate
    c_candidates = [0.1, 1.0, 10.0]
    results = []

    print("\n" + "-" * 65)
    print("TRAINING AND EVALUATING CANDIDATES ON VALIDATION SET")
    print("-" * 65)

    for c in c_candidates:
        print(f"\n>>> Evaluating LinearSVC with C = {c} (random_state=42, max_iter=2000)...")
        model = HOGLinearSVM(C=c, random_state=42, max_iter=2000, hog_config=HOG_CONFIG)

        t_start = time.time()
        model.fit(X_train, y_train)
        train_runtime = time.time() - t_start

        val_pred = model.predict(X_val)
        metrics = evaluate_predictions(y_val, val_pred)

        res_entry = {
            "C": c,
            "Accuracy": round(metrics["accuracy"], 6),
            "Macro Precision": round(metrics["macro_precision"], 6),
            "Macro Recall": round(metrics["macro_recall"], 6),
            "Macro F1": round(metrics["macro_f1"], 6),
            "Training Runtime": round(train_runtime, 2),
            "Confusion Matrix": metrics["confusion_matrix"].tolist(),
        }
        results.append(res_entry)

        print(f"    Train runtime:   {train_runtime:.2f}s")
        print(f"    Val Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"    Val Macro P:     {metrics['macro_precision']:.4f}")
        print(f"    Val Macro R:     {metrics['macro_recall']:.4f}")
        print(f"    Val Macro F1:    {metrics['macro_f1']:.4f}")

    # 4. Save results table
    table_csv_path = TABLES_DIR / "hog_svm_c_tuning_validation.csv"
    table_json_path = TABLES_DIR / "hog_svm_c_tuning_validation.json"

    with open(table_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["C", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Training Runtime"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    with open(table_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved validation results table to: {table_csv_path}")

    # 5. Model Selection based on Validation Macro F1
    best_candidate = max(results, key=lambda x: x["Macro F1"])

    print("\n" + "=" * 65)
    print("VALIDATION TUNING SUMMARY")
    print("=" * 65)
    print(f"{'C':<8} | {'Accuracy':<10} | {'Macro P':<10} | {'Macro R':<10} | {'Macro F1':<10} | {'Runtime (s)':<12}")
    print("-" * 65)
    for r in results:
        print(f"{r['C']:<8} | {r['Accuracy']:<10.4f} | {r['Macro Precision']:<10.4f} | {r['Macro Recall']:<10.4f} | {r['Macro F1']:<10.4f} | {r['Training Runtime']:<12.2f}")
    print("-" * 65)
    print(f"\nSelected Candidate by Validation Macro F1: C = {best_candidate['C']} (Macro F1 = {best_candidate['Macro F1']:.4f})")
    print("Test set status: UNTOUCHED (NOT loaded or evaluated in this experiment).")


if __name__ == "__main__":
    run_c_tuning()
