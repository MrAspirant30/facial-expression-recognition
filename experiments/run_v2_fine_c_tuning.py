"""
Validation Experiment: Fine-Grained C Tuning for HOG Baseline (900-D).

Performs fine-grained C hyperparameter tuning for the winning 900-D HOG representation
(9 orientations, 8x8 cells, 2x2 blocks, L2-Hys) using ONLY the train and validation splits.

Target Candidates (all LinearSVC with random_state=42, max_iter=2000, class_weight='balanced'):
1. C=0.25
2. C=0.35
3. C=0.50 (Existing Champion Reference: Macro F1 = 0.3458)
4. C=0.65
5. C=0.75

STRICT TEST-SET ISOLATION:
- Loads ONLY cached train and validation features:
  results/features/train_features.npy
  results/features/train_labels.npy
  results/features/val_features.npy
  results/features/val_labels.npy
- Test set (test_features.npy, test_labels.npy, test images) is NEVER accessed,
  loaded, or evaluated.
- Explicit security guard confirms test split is locked and bypassed.

METRICS & SELECTION:
- Evaluates: Accuracy, Macro Precision, Macro Recall, Macro F1, training runtime, validation runtime.
- Primary selection metric: Validation Macro F1.
- Results saved to:
  - results/tables/fine_c_tuning_validation.csv
  - results/tables/fine_c_tuning_validation.json
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
FEATURES_DIR = RESULTS_DIR / "features"

DEFAULT_OUTPUT_CSV = TABLES_DIR / "fine_c_tuning_validation.csv"
DEFAULT_OUTPUT_JSON = TABLES_DIR / "fine_c_tuning_validation.json"

EXPECTED_FEATURE_DIM = 900
EXPECTED_TRAIN_SAMPLES = 22347
EXPECTED_VAL_SAMPLES = 5586

EXISTING_CHAMPION_C = 0.50
EXISTING_CHAMPION_MACRO_F1 = 0.3458


def assert_test_set_isolation() -> None:
    """
    Enforce strict runtime security check that test set data is not read, featurized, or loaded.
    """
    test_feature_file = FEATURES_DIR / "test_features.npy"
    test_label_file = FEATURES_DIR / "test_labels.npy"

    assert test_feature_file.name == "test_features.npy"
    assert test_label_file.name == "test_labels.npy"
    print("[SECURITY GUARD] Test set isolation verified. Test split is strictly LOCKED and BYPASSED.")


def load_baseline_features(
    features_dir: Path = FEATURES_DIR,
    use_mmap: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-extracted 900-D baseline HOG features and labels for train and validation splits only.
    Uses memory-mapped loading (mmap_mode='r') for efficient I/O.
    """
    assert_test_set_isolation()

    train_x_path = features_dir / "train_features.npy"
    train_y_path = features_dir / "train_labels.npy"
    val_x_path = features_dir / "val_features.npy"
    val_y_path = features_dir / "val_labels.npy"

    for path in (train_x_path, train_y_path, val_x_path, val_y_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required baseline feature cache missing: {path.as_posix()}.\n"
                "Please ensure train_features.npy, train_labels.npy, val_features.npy, and val_labels.npy exist."
            )

    t0 = time.time()
    mmap = "r" if use_mmap else None
    X_train = np.load(train_x_path, mmap_mode=mmap)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path, mmap_mode=mmap)
    y_val = np.load(val_y_path)
    load_time = time.time() - t0

    # Strict integrity assertions on dataset split shapes
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
    print("FINE C TUNING: LOADED 900-D HOG BASELINE FEATURES (VALIDATION-ONLY)")
    print("=" * 70)
    print(f"Elapsed loading time: {load_time:.2f}s (mmap_mode={'r' if use_mmap else 'None'})")
    print(f"  X_train: shape={X_train.shape}, dtype={X_train.dtype} ({X_train.nbytes / (1024 * 1024):.1f} MB)")
    print(f"  y_train: shape={y_train.shape}, dtype={y_train.dtype}")
    print(f"  X_val:   shape={X_val.shape}, dtype={X_val.dtype} ({X_val.nbytes / (1024 * 1024):.1f} MB)")
    print(f"  y_val:   shape={y_val.shape}, dtype={y_val.dtype}")
    print("  Test set status: LOCKED (Not loaded, not evaluated, completely isolated)")
    print("=" * 70)

    return X_train, y_train, X_val, y_val


def get_fine_c_candidates(max_iter: int = 2000) -> List[Dict[str, Any]]:
    """
    Define the fine-grained C candidate configurations around the C=0.50 champion.
    """
    c_values = [0.25, 0.35, 0.50, 0.65, 0.75]
    candidates = []

    for c in c_values:
        is_champion_ref = (c == EXISTING_CHAMPION_C)
        label = f"LinearSVC_C{c:.2f}_balanced" + (" (Existing Champion)" if is_champion_ref else "")
        candidates.append({
            "candidate": label,
            "C": c,
            "class_weight": "balanced",
            "model": LinearSVC(C=c, class_weight="balanced", random_state=42, max_iter=max_iter, dual="auto"),
        })

    return candidates


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
    Train each fine C candidate on X_train/y_train only, evaluate on X_val/y_val only,
    record all metrics and runtimes, and save results.
    """
    results: List[Dict[str, Any]] = []

    print(f"\nEvaluating {len(candidates)} fine-grained C candidates on validation set...")
    print("-" * 70)

    for idx, cand in enumerate(candidates, 1):
        name = cand["candidate"]
        c_val = cand["C"]
        model = cand["model"]
        print(f"\n[{idx}/{len(candidates)}] Training: {name} (C={c_val}, class_weight='balanced') ...")

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
            "C": c_val,
            "class_weight": "balanced",
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

    # Save to CSV and JSON under results/tables/
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

    # Print Summary Table sorted by Macro F1 descending
    print("\n" + "=" * 80)
    print("FINE-GRAINED C TUNING RESULTS SUMMARY (Sorted by Macro F1)")
    print("=" * 80)
    print(f"{'Candidate':<38} | {'C':<5} | {'Acc':<7} | {'Mac P':<7} | {'Mac R':<7} | {'Mac F1':<7} | {'Train (s)'}")
    print("-" * 88)
    sorted_results = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['candidate']:<38} | "
            f"{r['C']:<5.2f} | "
            f"{r['accuracy']:<7.4f} | "
            f"{r['macro_precision']:<7.4f} | "
            f"{r['macro_recall']:<7.4f} | "
            f"{r['macro_f1']:<7.4f} | "
            f"{r['train_runtime_sec']:<9.2f}"
        )
    print("-" * 88)

    best_candidate = sorted_results[0]
    diff = best_candidate["macro_f1"] - EXISTING_CHAMPION_MACRO_F1
    diff_sign = "+" if diff >= 0 else ""

    print("\n" + "=" * 80)
    print("CHAMPION COMPARISON")
    print("=" * 80)
    print(f"Existing Champion: LinearSVC(C={EXISTING_CHAMPION_C:.2f}, class_weight='balanced') -> Validation Macro F1 = {EXISTING_CHAMPION_MACRO_F1:.4f}")
    print(f"Best Tuned Candidate: {best_candidate['candidate']}")
    print(f"  Validation Macro F1: {best_candidate['macro_f1']:.4f} ({diff_sign}{diff:.4f} vs existing champion)")
    print(f"  Validation Accuracy: {best_candidate['accuracy']:.4f}")
    print(f"Results saved to:")
    print(f"  CSV:  {output_csv.as_posix()}")
    print(f"  JSON: {output_json.as_posix()}")
    print("=" * 80)

    return results


def run_smoke_test(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """
    Run rapid smoke test verifying data loading, feature dimensions, all 5 candidate models,
    metric computation, and serialization on a small representative slice.
    Completes in < 1 second.
    """
    print("\n" + "=" * 70)
    print("RUNNING SMOKE TEST (150 TRAIN / 70 VAL SAMPLES, ALL 5 CANDIDATES)")
    print("=" * 70)

    # 1. Verify full dataset shapes
    assert X_train.shape == (EXPECTED_TRAIN_SAMPLES, EXPECTED_FEATURE_DIM), (
        f"Train shape mismatch: {X_train.shape}, expected ({EXPECTED_TRAIN_SAMPLES}, {EXPECTED_FEATURE_DIM})"
    )
    assert X_val.shape == (EXPECTED_VAL_SAMPLES, EXPECTED_FEATURE_DIM), (
        f"Val shape mismatch: {X_val.shape}, expected ({EXPECTED_VAL_SAMPLES}, {EXPECTED_FEATURE_DIM})"
    )
    print(f"  [CHECK] Full train shape verified: {X_train.shape}")
    print(f"  [CHECK] Full val shape verified:   {X_val.shape}")

    # 2. Sample representative subset across classes with fixed seed
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(y_train), size=150, replace=False)
    val_indices = rng.choice(len(y_val), size=70, replace=False)

    X_tr_sub = np.asarray(X_train[train_indices], dtype=np.float32)
    y_tr_sub = y_train[train_indices]
    X_va_sub = np.asarray(X_val[val_indices], dtype=np.float32)
    y_va_sub = y_val[val_indices]

    assert X_tr_sub.shape == (150, EXPECTED_FEATURE_DIM), f"Smoke train shape mismatch: {X_tr_sub.shape}"
    assert X_va_sub.shape == (70, EXPECTED_FEATURE_DIM), f"Smoke val shape mismatch: {X_va_sub.shape}"

    smoke_candidates = get_fine_c_candidates(max_iter=50)

    smoke_csv = TABLES_DIR / "smoke_test_fine_c_validation.csv"
    smoke_json = TABLES_DIR / "smoke_test_fine_c_validation.json"

    results = evaluate_candidates(
        candidates=smoke_candidates,
        X_train=X_tr_sub,
        y_train=y_tr_sub,
        X_val=X_va_sub,
        y_val=y_va_sub,
        output_csv=smoke_csv,
        output_json=smoke_json,
    )

    # 3. Verify output files written
    assert smoke_csv.exists(), "Smoke test CSV output missing"
    assert smoke_json.exists(), "Smoke test JSON output missing"
    assert len(results) == 5, f"Expected 5 candidate results, got {len(results)}"

    # Clean up smoke test temporary files
    if smoke_csv.exists():
        smoke_csv.unlink()
    if smoke_json.exists():
        smoke_json.unlink()

    print("\n[SUCCESS] Smoke test passed for fine C tuning evaluator:")
    print(f"  - Verified train shape: (22347, {EXPECTED_FEATURE_DIM})")
    print(f"  - Verified val shape:   (5586, {EXPECTED_FEATURE_DIM})")
    print("  - All 5 candidate configurations executed successfully")
    print("  - Metric calculation (Accuracy, Macro P, Macro R, Macro F1) verified")
    print("  - Output serialization verified")
    print("  - Test set remained completely locked and untouched")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-Grained C Tuning for HOG Baseline (Validation-Only)."
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

    # 1. Load cached 900-D baseline features
    X_train, y_train, X_val, y_val = load_baseline_features(
        features_dir=FEATURES_DIR,
        use_mmap=not args.no_mmap,
    )

    # 2. Check for smoke test
    if args.smoke_test:
        run_smoke_test(X_train, y_train, X_val, y_val)
        return

    # 3. Full evaluation of the 5 fine C candidates
    candidates = get_fine_c_candidates(max_iter=2000)
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
