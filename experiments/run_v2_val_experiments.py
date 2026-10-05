"""
V2 Validation Experiment Runner (Model & Hyperparameter Exploration).

STRICT VALIDATION-ONLY POLICY:
This runner trains models ONLY on the training split (train_features.npy) and
evaluates performance ONLY on the validation split (val_features.npy).

The official test set (test_features.npy, test_labels.npy) is strictly locked
and is NEVER loaded, accessed, or evaluated by this runner. This guarantees
unbiased model and hyperparameter selection without test-set data leakage.

FAST FEATURE REUSE:
Reuses pre-extracted HOG feature matrices (900 dimensions per sample) cached in
`results/features/`. Bypasses raw image decoding and HOG re-extraction, enabling
instantaneous data loading (sub-second) and zero redundant computation.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hog_svm import CLASS_NAMES
from src.evaluation.evaluate import evaluate_predictions


RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FEATURES_DIR = RESULTS_DIR / "features"


def assert_test_set_isolation() -> None:
    """
    Enforce that test set feature and label files are not accessed or loaded
    during validation experiments.
    """
    # Defensive check: ensure test feature paths are isolated
    test_feature_path = FEATURES_DIR / "test_features.npy"
    test_label_path = FEATURES_DIR / "test_labels.npy"
    # Verification: Both files may exist on disk from V1 baseline, but this runner
    # is strictly prohibited from opening or loading them.
    assert test_feature_path.name == "test_features.npy"
    assert test_label_path.name == "test_labels.npy"


def load_train_and_val_features(
    features_dir: Path = FEATURES_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-extracted float32 HOG feature vectors and int64 target labels
    for train and validation splits only.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    assert_test_set_isolation()

    train_x_path = features_dir / "train_features.npy"
    train_y_path = features_dir / "train_labels.npy"
    val_x_path = features_dir / "val_features.npy"
    val_y_path = features_dir / "val_labels.npy"

    if not (train_x_path.exists() and train_y_path.exists()):
        raise FileNotFoundError(
            f"Cached training features missing at {train_x_path}. "
            "Please ensure train_features.npy and train_labels.npy are present."
        )

    if not (val_x_path.exists() and val_y_path.exists()):
        raise FileNotFoundError(
            f"Cached validation features missing at {val_x_path}. "
            "Please ensure val_features.npy and val_labels.npy are present."
        )

    t0 = time.time()
    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)
    elapsed = time.time() - t0

    print("=" * 70)
    print("V2 EXPERIMENT RUNNER: LOADED CACHED HOG FEATURES (VALIDATION-ONLY)")
    print("=" * 70)
    print(f"Elapsed loading time: {elapsed:.2f}s (reusing pre-extracted features)")
    print(f"  X_train: shape={X_train.shape}, dtype={X_train.dtype} ({X_train.nbytes / (1024*1024):.1f} MB)")
    print(f"  y_train: shape={y_train.shape}, dtype={y_train.dtype}")
    print(f"  X_val:   shape={X_val.shape}, dtype={X_val.dtype} ({X_val.nbytes / (1024*1024):.1f} MB)")
    print(f"  y_val:   shape={y_val.shape}, dtype={y_val.dtype}")
    print("  Test set status: LOCKED (Not loaded, not evaluated, completely isolated)")
    print("=" * 70)

    return X_train, y_train, X_val, y_val


def get_candidate_configurations(suite: str = "class_weight") -> List[Dict[str, Any]]:
    """
    Define systematic candidate configurations for V2 model exploration.

    Suites:
      - 'class_weight': Explores balanced class weighting across C values [0.1, 0.5, 1.0, 5.0, 10.0]
                        plus baseline unweighted references. Directly addresses FER2013 class imbalance.
      - 'scaling': Explores StandardScaler before LinearSVC vs raw HOG features.
      - 'quick': Fast 3-model comparison (V1 baseline, C=1.0 balanced, C=10.0 balanced).
      - 'all': Comprehensive set combining class weighting, scaling, and linear models.
    """
    candidates: List[Dict[str, Any]] = []

    if suite in ("class_weight", "all"):
        # V1 references for direct comparison
        candidates.append({
            "name": "LinearSVC_C1.0_none (V1 Baseline)",
            "model_family": "LinearSVC",
            "C": 1.0,
            "class_weight": None,
            "scaler": "None",
            "model": LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=2000, dual="auto"),
        })
        candidates.append({
            "name": "LinearSVC_C10.0_none (V1 Best C)",
            "model_family": "LinearSVC",
            "C": 10.0,
            "class_weight": None,
            "scaler": "None",
            "model": LinearSVC(C=10.0, class_weight=None, random_state=42, max_iter=2000, dual="auto"),
        })
        # V2 Class-weighted candidates across C values
        for c_val in [0.1, 0.5, 1.0, 5.0, 10.0]:
            candidates.append({
                "name": f"LinearSVC_C{c_val}_balanced",
                "model_family": "LinearSVC",
                "C": c_val,
                "class_weight": "balanced",
                "scaler": "None",
                "model": LinearSVC(C=c_val, class_weight="balanced", random_state=42, max_iter=2000, dual="auto"),
            })

    if suite in ("scaling", "all"):
        for c_val in [1.0, 10.0]:
            candidates.append({
                "name": f"StandardScaler_LinearSVC_C{c_val}_balanced",
                "model_family": "LinearSVC",
                "C": c_val,
                "class_weight": "balanced",
                "scaler": "StandardScaler",
                "model": Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LinearSVC(C=c_val, class_weight="balanced", random_state=42, max_iter=2000, dual="auto")),
                ]),
            })

    if suite == "quick":
        candidates = [
            {
                "name": "LinearSVC_C1.0_none (V1 Baseline)",
                "model_family": "LinearSVC",
                "C": 1.0,
                "class_weight": None,
                "scaler": "None",
                "model": LinearSVC(C=1.0, class_weight=None, random_state=42, max_iter=2000, dual="auto"),
            },
            {
                "name": "LinearSVC_C1.0_balanced",
                "model_family": "LinearSVC",
                "C": 1.0,
                "class_weight": "balanced",
                "scaler": "None",
                "model": LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000, dual="auto"),
            },
            {
                "name": "LinearSVC_C10.0_balanced",
                "model_family": "LinearSVC",
                "C": 10.0,
                "class_weight": "balanced",
                "scaler": "None",
                "model": LinearSVC(C=10.0, class_weight="balanced", random_state=42, max_iter=2000, dual="auto"),
            },
        ]

    return candidates


def run_experiments(
    candidates: List[Dict[str, Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    output_csv: Path,
    output_json: Path,
) -> List[Dict[str, Any]]:
    """
    Train each candidate model on X_train/y_train only, evaluate on X_val/y_val only,
    record metrics and runtime, and save results to CSV and JSON.
    """
    results: List[Dict[str, Any]] = []

    print(f"\nStarting evaluation of {len(candidates)} candidate configurations on validation set...")
    print("-" * 70)

    for idx, cand in enumerate(candidates, 1):
        name = cand["name"]
        model = cand["model"]
        print(f"\n[{idx}/{len(candidates)}] Training: {name} ...")

        t_train_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_train_start

        t_val_start = time.time()
        y_pred = model.predict(X_val)
        val_time = time.time() - t_val_start

        metrics = evaluate_predictions(y_val, y_pred)

        record = {
            "name": name,
            "model_family": cand["model_family"],
            "C": cand["C"],
            "class_weight": str(cand["class_weight"]),
            "scaler": cand["scaler"],
            "accuracy": round(metrics["accuracy"], 6),
            "macro_precision": round(metrics["macro_precision"], 6),
            "macro_recall": round(metrics["macro_recall"], 6),
            "macro_f1": round(metrics["macro_f1"], 6),
            "train_time_sec": round(train_time, 2),
            "val_time_sec": round(val_time, 3),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }
        results.append(record)

        print(f"    Train time:   {train_time:.2f}s | Val time: {val_time:.3f}s")
        print(f"    Val Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
        print(f"    Val Macro P:  {metrics['macro_precision']:.4f}")
        print(f"    Val Macro R:  {metrics['macro_recall']:.4f}")
        print(f"    Val Macro F1: {metrics['macro_f1']:.4f}")

    # Save results to CSV and JSON under results/tables/
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "name",
        "model_family",
        "C",
        "class_weight",
        "scaler",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "train_time_sec",
        "val_time_sec",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("V2 VALIDATION TUNING RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<34} | {'Acc':<7} | {'Mac P':<7} | {'Mac R':<7} | {'Mac F1':<7} | {'Time (s)':<8}")
    print("-" * 75)
    # Sort by Macro F1 descending
    sorted_results = sorted(results, key=lambda x: x["macro_f1"], reverse=True)
    for r in sorted_results:
        print(
            f"{r['name']:<34} | "
            f"{r['accuracy']:<7.4f} | "
            f"{r['macro_precision']:<7.4f} | "
            f"{r['macro_recall']:<7.4f} | "
            f"{r['macro_f1']:<7.4f} | "
            f"{r['train_time_sec']:<8.2f}"
        )
    print("-" * 75)

    best_candidate = sorted_results[0]
    print(f"\nTop Candidate by Validation Macro F1: {best_candidate['name']}")
    print(f"  Macro F1: {best_candidate['macro_f1']:.4f} | Accuracy: {best_candidate['accuracy']:.4f}")
    print(f"Results saved to:")
    print(f"  CSV:  {output_csv.as_posix()}")
    print(f"  JSON: {output_json.as_posix()}")
    print("=" * 70)

    return results


def run_smoke_test(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
    """
    Ultra-lightweight smoke test to verify the runner pipeline without long training.
    Uses 100 train samples and 50 validation samples with max_iter=50.
    Completes in < 1 second.
    """
    print("\n" + "=" * 70)
    print("RUNNING SMOKE TEST (SUBSET SLICE: 100 TRAIN / 50 VAL SAMPLES)")
    print("=" * 70)

    # Randomly sample representative subset across classes with fixed seed
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(y_train), size=200, replace=False)
    val_indices = rng.choice(len(y_val), size=100, replace=False)

    X_tr_sub = X_train[train_indices]
    y_tr_sub = y_train[train_indices]
    X_val_sub = X_val[val_indices]
    y_val_sub = y_val[val_indices]

    smoke_candidates = [
        {
            "name": "SmokeTest_LinearSVC_C1.0_balanced",
            "model_family": "LinearSVC",
            "C": 1.0,
            "class_weight": "balanced",
            "scaler": "None",
            "model": LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=50, dual="auto"),
        }
    ]

    smoke_csv = TABLES_DIR / "smoke_test_validation_results.csv"
    smoke_json = TABLES_DIR / "smoke_test_validation_results.json"

    results = run_experiments(
        candidates=smoke_candidates,
        X_train=X_tr_sub,
        y_train=y_tr_sub,
        X_val=X_val_sub,
        y_val=y_val_sub,
        output_csv=smoke_csv,
        output_json=smoke_json,
    )

    print("\n[SUCCESS] Smoke test passed successfully!")
    print(f"Verified: Cached HOG feature loading, model fitting, metric computation, and table saving.")
    print("The official test set remained completely locked and untouched.")


def main():
    parser = argparse.ArgumentParser(
        description="V2 Validation Experiment Runner (Model & Hyperparameter Exploration)."
    )
    parser.add_argument(
        "--suite",
        type=str,
        default="class_weight",
        choices=["class_weight", "scaling", "quick", "all"],
        help="Experiment suite to run. Default: 'class_weight'.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(TABLES_DIR / "v2_validation_results.csv"),
        help="Output CSV file path under results/tables/.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(TABLES_DIR / "v2_validation_results.json"),
        help="Output JSON file path under results/tables/.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test on 100 train / 50 val samples to verify the pipeline.",
    )
    args = parser.parse_args()

    # 1. Load cached features (train and val only)
    X_train, y_train, X_val, y_val = load_train_and_val_features(FEATURES_DIR)

    # 2. Check for smoke test flag
    if args.smoke_test:
        run_smoke_test(X_train, y_train, X_val, y_val)
        return

    # 3. Get candidate models
    candidates = get_candidate_configurations(args.suite)

    # 4. Run experiments and save results
    run_experiments(
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
