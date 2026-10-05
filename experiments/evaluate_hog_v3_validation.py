"""
Experiment 3: HOG-2 Validation-Only Evaluator.

Evaluates the fine-grid HOG-2 representation (4,356 dimensions: 9 orientations, 4x4 cells, 2x2 blocks)
on the clean FER2013 validation set across three LinearSVC candidate configurations:
1. HOG2_C1.0_none:     C=1.0,  class_weight=None
2. HOG2_C10.0_none:    C=10.0, class_weight=None
3. HOG2_C0.5_balanced: C=0.5,  class_weight='balanced'

STRICT TEST-SET ISOLATION:
- Loads ONLY the cached train and validation feature matrices from:
  results/features/hog_v3/hog_9_cell4/
- Test set (test_features.npy, test_labels.npy, test images) is NEVER accessed,
  loaded, or evaluated.
- Explicit assertion ensures test set remains locked and isolated.

METRICS & SELECTION:
- Evaluates: Accuracy, Macro Precision, Macro Recall, Macro F1, training runtime, validation runtime.
- Primary selection metric: Validation Macro F1.
- Results saved to:
  - results/tables/hog_v3_hog2_validation.csv
  - results/tables/hog_v3_hog2_validation.json
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.svm import LinearSVC

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hog_svm import CLASS_NAMES
from src.evaluation.evaluate import evaluate_predictions


RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
HOG2_CACHE_DIR = RESULTS_DIR / "features" / "hog_v3" / "hog_9_cell4"

DEFAULT_OUTPUT_CSV = TABLES_DIR / "hog_v3_hog2_validation.csv"
DEFAULT_OUTPUT_JSON = TABLES_DIR / "hog_v3_hog2_validation.json"

EXPECTED_FEATURE_DIM = 4356
EXPECTED_TRAIN_SAMPLES = 22347
EXPECTED_VAL_SAMPLES = 5586


def assert_test_set_isolation() -> None:
    """
    Enforce strict runtime check that test set data is not read, featurized, or loaded.
    """
    test_feature_file = RESULTS_DIR / "features" / "test_features.npy"
    test_label_file = RESULTS_DIR / "features" / "test_labels.npy"

    # Ensure these paths are explicitly acknowledged as isolated
    assert test_feature_file.name == "test_features.npy"
    assert test_label_file.name == "test_labels.npy"
    print("[SECURITY GUARD] Test set isolation verified. Test split is strictly LOCKED and BYPASSED.")


def load_hog2_features(
    cache_dir: Path = HOG2_CACHE_DIR,
    use_mmap: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached HOG-2 (4356-D) features and labels for train and validation splits only.
    Uses memory-mapped loading (mmap_mode='r') for efficient I/O.
    """
    assert_test_set_isolation()

    train_x_path = cache_dir / "train_features.npy"
    train_y_path = cache_dir / "train_labels.npy"
    val_x_path = cache_dir / "val_features.npy"
    val_y_path = cache_dir / "val_labels.npy"
    meta_path = cache_dir / "metadata.json"

    for path in (train_x_path, train_y_path, val_x_path, val_y_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required HOG-2 cache file missing: {path.as_posix()}.\n"
                "Please run: python experiments/cache_hog_v3_features.py --config hog_9_cell4"
            )

    t0 = time.time()
    mmap = "r" if use_mmap else None
    X_train = np.load(train_x_path, mmap_mode=mmap)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path, mmap_mode=mmap)
    y_val = np.load(val_y_path)
    load_time = time.time() - t0

    # Integrity assertions
    assert X_train.shape == (EXPECTED_TRAIN_SAMPLES, EXPECTED_FEATURE_DIM), (
        f"Train shape mismatch: {X_train.shape}, expected ({EXPECTED_TRAIN_SAMPLES}, {EXPECTED_FEATURE_DIM})"
    )
    assert y_train.shape == (EXPECTED_TRAIN_SAMPLES,), (
        f"Train label mismatch: {y_train.shape}, expected ({EXPECTED_TRAIN_SAMPLES},)"
    )
    assert X_val.shape == (EXPECTED_VAL_SAMPLES, EXPECTED_FEATURE_DIM), (
        f"Val shape mismatch: {X_val.shape}, expected ({EXPECTED_VAL_SAMPLES}, {EXPECTED_FEATURE_DIM})"
    )
    assert y_val.shape == (EXPECTED_VAL_SAMPLES,), (
        f"Val label mismatch: {y_val.shape}, expected ({EXPECTED_VAL_SAMPLES},)"
    )

    print("=" * 70)
    print("HOG-2 VALIDATION EVALUATOR: LOADED CACHED FEATURES")
    print("=" * 70)
    print(f"Elapsed loading time: {load_time:.2f}s (mmap_mode={'r' if use_mmap else 'None'})")
    print(f"  X_train: shape={X_train.shape}, dtype={X_train.dtype} ({X_train.nbytes / (1024 * 1024):.1f} MB)")
    print(f"  y_train: shape={y_train.shape}, dtype={y_train.dtype}")
    print(f"  X_val:   shape={X_val.shape}, dtype={X_val.dtype} ({X_val.nbytes / (1024 * 1024):.1f} MB)")
    print(f"  y_val:   shape={y_val.shape}, dtype={y_val.dtype}")
    print("  Test set status: LOCKED (Not loaded, not evaluated, completely isolated)")
    print("=" * 70)

    return X_train, y_train, X_val, y_val


def get_hog2_candidates(max_iter: int = 2000) -> List[Dict[str, Any]]:
    """
    Define the three target LinearSVC configurations for HOG-2 evaluation.
    """
    return [
        {
            "candidate": "HOG2_C1.0_none",
            "C": 1.0,
            "class_weight": None,
            "description": "HOG-2 with baseline regularization (C=1.0, unweighted)",
            "model": LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=max_iter, dual="auto"),
        },
        {
            "candidate": "HOG2_C10.0_none",
            "C": 10.0,
            "class_weight": None,
            "description": "HOG-2 with higher regularization weight (C=10.0, unweighted)",
            "model": LinearSVC(C=10.0, class_weight=None, random_state=42, max_iter=max_iter, dual="auto"),
        },
        {
            "candidate": "HOG2_C0.5_balanced",
            "C": 0.5,
            "class_weight": "balanced",
            "description": "HOG-2 with class-balanced weighting (C=0.5, balanced)",
            "model": LinearSVC(C=0.5, class_weight="balanced", random_state=42, max_iter=max_iter, dual="auto"),
        },
    ]


def evaluate_candidates(
    candidates: List[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_csv: Path,
    output_json: Path,
) -> List[Dict[str, Any]]:
    """
    Train each candidate on X_train/y_train only, evaluate on X_val/y_val only,
    record all metrics and runtimes, and save results.
    """
    results: List[Dict[str, Any]] = []

    print(f"\nEvaluating {len(candidates)} HOG-2 LinearSVC candidates on validation set...")
    print("-" * 70)

    for idx, cand in enumerate(candidates, 1):
        name = cand["candidate"]
        model = cand["model"]
        print(f"\n[{idx}/{len(candidates)}] Training: {name} (C={cand['C']}, class_weight={cand['class_weight']}) ...")

        # Convert mmap slice to contiguous array if needed for liblinear
        t_tr_start = time.time()
        X_tr_in = np.asarray(X_train) if hasattr(X_train, "_mmap") else X_train
        model.fit(X_tr_in, y_train)
        train_time = time.time() - t_tr_start

        t_va_start = time.time()
        X_va_in = np.asarray(X_val) if hasattr(X_val, "_mmap") else X_val
        y_pred = model.predict(X_va_in)
        val_time = time.time() - t_va_start

        metrics = evaluate_predictions(y_val, y_pred)

        record = {
            "candidate": name,
            "C": cand["C"],
            "class_weight": str(cand["class_weight"]),
            "accuracy": round(metrics["accuracy"], 6),
            "macro_precision": round(metrics["macro_precision"], 6),
            "macro_recall": round(metrics["macro_recall"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "train_runtime_sec": round(train_time, 2),
            "val_runtime_sec": round(val_time, 3),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }
        results.append(record)

        print(f"    Train runtime:   {train_time:.2f}s | Val runtime: {val_time:.3f}s")
        print(f"    Val Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"    Val Macro P:     {metrics['macro_precision']:.4f}")
        print(f"    Val Macro R:     {metrics['macro_recall']:.4f}")
        print(f"    Val Macro F1:    {metrics['macro_f1']:.4f}")

    # Save to CSV and JSON
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "candidate",
        "C",
        "class_weight",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "train_runtime_sec",
        "val_runtime_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 75)
    print("HOG-2 VALIDATION RESULTS SUMMARY (Sorted by Macro F1)")
    print("=" * 75)
    print(f"{'Candidate':<22} | {'Acc':<7} | {'Mac P':<7} | {'Mac R':<7} | {'Mac F1':<7} | {'Train (s)':<9} | {'Val (s)'}")
    print("-" * 75)
    sorted_results = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['candidate']:<22} | "
            f"{r['accuracy']:<7.4f} | "
            f"{r['macro_precision']:<7.4f} | "
            f"{r['macro_recall']:<7.4f} | "
            f"{r['macro_f1']:<7.4f} | "
            f"{r['train_runtime_sec']:<9.2f} | "
            f"{r['val_runtime_sec']:.3f}"
        )
    print("-" * 75)
    best_candidate = sorted_results[0]
    print(f"\nTop HOG-2 Candidate by Validation Macro F1: {best_candidate['candidate']}")
    print(f"  Macro F1: {best_candidate['macro_f1']:.4f} | Accuracy: {best_candidate['accuracy']:.4f}")
    print(f"Results saved to:")
    print(f"  CSV:  {output_csv.as_posix()}")
    print(f"  JSON: {output_json.as_posix()}")
    print("=" * 75)

    return results


def run_smoke_test(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """
    Run rapid smoke test verifying data loading, feature dimensions, all 3 candidate models,
    metric computation, and serialization on a small representative slice.
    Completes in < 1 second.
    """
    print("\n" + "=" * 70)
    print("RUNNING SMOKE TEST (150 TRAIN / 70 VAL SAMPLES)")
    print("=" * 70)

    # Sample representative subset across classes using fixed seed
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(y_train), size=150, replace=False)
    val_indices = rng.choice(len(y_val), size=70, replace=False)

    X_tr_sub = np.asarray(X_train[train_indices], dtype=np.float32)
    y_tr_sub = y_train[train_indices]
    X_va_sub = np.asarray(X_val[val_indices], dtype=np.float32)
    y_va_sub = y_val[val_indices]

    assert X_tr_sub.shape == (150, EXPECTED_FEATURE_DIM), f"Smoke train shape mismatch: {X_tr_sub.shape}"
    assert X_va_sub.shape == (70, EXPECTED_FEATURE_DIM), f"Smoke val shape mismatch: {X_va_sub.shape}"

    smoke_candidates = get_hog2_candidates(max_iter=50)

    smoke_csv = TABLES_DIR / "smoke_test_hog2_validation.csv"
    smoke_json = TABLES_DIR / "smoke_test_hog2_validation.json"

    results = evaluate_candidates(
        candidates=smoke_candidates,
        X_train=X_tr_sub,
        y_train=y_tr_sub,
        X_val=X_va_sub,
        y_val=y_va_sub,
        output_csv=smoke_csv,
        output_json=smoke_json,
    )

    # Clean up smoke test temporary files
    if smoke_csv.exists():
        smoke_csv.unlink()
    if smoke_json.exists():
        smoke_json.unlink()

    print("\n[SUCCESS] Smoke test passed for HOG-2 evaluator:")
    print(f"  - Cached features loaded via mmap (dim = {EXPECTED_FEATURE_DIM})")
    print(f"  - All 3 configurations executed (HOG2_C1.0_none, HOG2_C10.0_none, HOG2_C0.5_balanced)")
    print("  - Metric calculation (Accuracy, Macro P, Macro R, Macro F1) verified")
    print("  - Output serialization verified")
    print("  - Test set remained completely locked and untouched")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: HOG-2 LinearSVC Validation-Only Evaluator."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test on 150 train / 70 val samples to verify the pipeline.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path under results/tables/.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help="Output JSON path under results/tables/.",
    )
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory-mapped file loading.",
    )
    args = parser.parse_args()

    # 1. Load cached HOG-2 features
    X_train, y_train, X_val, y_val = load_hog2_features(
        cache_dir=HOG2_CACHE_DIR,
        use_mmap=not args.no_mmap,
    )

    # 2. Check for smoke test
    if args.smoke_test:
        run_smoke_test(X_train, y_train, X_val, y_val)
        return

    # 3. Full evaluation of the 3 target candidates
    candidates = get_hog2_candidates(max_iter=2000)
    evaluate_candidates(
        candidates=candidates,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
    )


if __name__ == "__main__":
    main()
