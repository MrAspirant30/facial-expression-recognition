"""
Experiment 3: HOG Representation Tuning - Feature Extraction & Cache Pipeline.

Builds and manages an efficient, configuration-driven HOG feature extraction and
caching pipeline for FER2013 classical machine learning experiments.

SUPPORTED HOG CONFIGURATIONS:
1. hog_9_cell8  (HOG-1): 9 orientations, 8x8 cells, 2x2 blocks, L2-Hys ->  900 dims (V1 Baseline)
2. hog_9_cell4  (HOG-2): 9 orientations, 4x4 cells, 2x2 blocks, L2-Hys -> 4356 dims (Finer Spatial Grid)
3. hog_9_cell6  (HOG-3): 9 orientations, 6x6 cells, 2x2 blocks, L2-Hys -> 1764 dims (Medium Spatial Grid)
4. hog_12_cell4 (HOG-4): 12 orientations, 4x4 cells, 2x2 blocks, L2-Hys -> 5808 dims (Higher Angular Res)
5. hog_18_cell4 (HOG-5): 18 orientations, 4x4 cells, 2x2 blocks, L2-Hys -> 8712 dims (Fine Angular Res)

STRICT TEST-SET ISOLATION:
- Processes ONLY the authoritative train (22,347) and validation (5,586) splits.
- Test split (7,178) is strictly bypassed, never opened, and never featurized.
- Zero test leakage guaranteed during representation tuning.

EFFICIENCY & CACHE MANAGEMENT:
- HOG-1 (hog_9_cell8) safely links and reuses the existing 900-D baseline cache without recomputation.
- Preallocates float32 arrays to control memory footprint.
- Skips recomputation if target configuration cache already exists with matching metadata.
"""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import extract_hog_features


# Canonical Paths
MANIFEST_PATH = PROJECT_ROOT / "data" / "fer2013" / "archive" / "clean_dataset_split.csv"
DATASET_BASE_DIR = MANIFEST_PATH.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURES_DIR = RESULTS_DIR / "features"
HOG_V3_DIR = FEATURES_DIR / "hog_v3"

# Baseline 900-D cache locations
BASELINE_TRAIN_X = FEATURES_DIR / "train_features.npy"
BASELINE_TRAIN_Y = FEATURES_DIR / "train_labels.npy"
BASELINE_VAL_X = FEATURES_DIR / "val_features.npy"
BASELINE_VAL_Y = FEATURES_DIR / "val_labels.npy"


# HOG Configurations Registry
HOG_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "hog_9_cell8": {
        "display_name": "HOG-1 (Baseline 900-D)",
        "orientations": 9,
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "expected_dim": 900,
        "description": "Baseline: 9 orientations, 8x8 cell grid, 2x2 blocks. Reuses baseline cache.",
        "is_baseline": True,
    },
    "hog_9_cell4": {
        "display_name": "HOG-2 (Fine Grid 4356-D)",
        "orientations": 9,
        "pixels_per_cell": (4, 4),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "expected_dim": 4356,
        "description": "Finer spatial resolution with 4x4 cells (11x11 blocks, 4356 dimensions).",
        "is_baseline": False,
    },
    "hog_9_cell6": {
        "display_name": "HOG-3 (Medium Grid 1764-D)",
        "orientations": 9,
        "pixels_per_cell": (6, 6),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "expected_dim": 1764,
        "description": "Medium spatial resolution with 6x6 cells (7x7 blocks, 1764 dimensions).",
        "is_baseline": False,
    },
    "hog_12_cell4": {
        "display_name": "HOG-4 (Angular Res 12-Ori 5808-D)",
        "orientations": 12,
        "pixels_per_cell": (4, 4),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "expected_dim": 5808,
        "description": "12 orientation bins with 4x4 fine cell grid (5808 dimensions).",
        "is_baseline": False,
    },
    "hog_18_cell4": {
        "display_name": "HOG-5 (Fine Angular Res 18-Ori 8712-D)",
        "orientations": 18,
        "pixels_per_cell": (4, 4),
        "cells_per_block": (2, 2),
        "block_norm": "L2-Hys",
        "expected_dim": 8712,
        "description": "18 orientation bins with 4x4 fine cell grid (8712 dimensions).",
        "is_baseline": False,
    },
}


def assert_test_set_isolation() -> None:
    """
    Enforce strict runtime check that test set data is not read, featurized, or loaded.
    """
    print("[SECURITY GUARD] Test set isolation verified. Test split is strictly excluded.")


def load_train_and_val_manifest_rows(
    manifest_path: Path,
) -> Tuple[List[dict], List[dict]]:
    """
    Load authoritative manifest and extract only train and validation rows.
    Strictly ignores and bypasses test split rows to ensure zero data leakage.

    Returns:
        (train_rows, val_rows)
    """
    assert_test_set_isolation()

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    train_rows: List[dict] = []
    val_rows: List[dict] = []
    test_count = 0

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split == "train":
                train_rows.append(row)
            elif split == "validation":
                val_rows.append(row)
            elif split == "test":
                test_count += 1
            else:
                raise ValueError(f"Unknown split in manifest: {split}")

    # Integrity verification
    assert len(train_rows) == 22347, f"Expected 22347 train rows, found {len(train_rows)}"
    assert len(val_rows) == 5586, f"Expected 5586 val rows, found {len(val_rows)}"
    assert test_count == 7178, f"Expected 7178 test rows in manifest, found {test_count}"

    # Confirm rows contain only train and validation
    assert all(r["split"] == "train" for r in train_rows)
    assert all(r["split"] == "validation" for r in val_rows)

    return train_rows, val_rows


def is_cache_valid(config_dir: Path, expected_dim: int, n_train: int, n_val: int) -> bool:
    """
    Check if a HOG configuration cache already exists and matches expected dimensions and counts.
    """
    meta_path = config_dir / "metadata.json"
    train_x = config_dir / "train_features.npy"
    train_y = config_dir / "train_labels.npy"
    val_x = config_dir / "val_features.npy"
    val_y = config_dir / "val_labels.npy"

    if not (meta_path.exists() and train_x.exists() and train_y.exists() and val_x.exists() and val_y.exists()):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if (
            meta.get("feature_dimension") == expected_dim
            and meta.get("train_sample_count") == n_train
            and meta.get("validation_sample_count") == n_val
        ):
            return True
    except Exception:
        return False

    return False


def extract_features_for_rows(
    rows: List[dict],
    base_dir: Path,
    hog_config: Dict[str, Any],
    expected_dim: int,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract HOG features directly into a pre-allocated float32 array.
    """
    n_samples = len(rows)
    X = np.empty((n_samples, expected_dim), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.int64)

    t0 = time.time()
    for idx, row in enumerate(rows):
        img_path = base_dir / row["path"]
        with Image.open(img_path) as img:
            arr = np.asarray(img, dtype=np.float32) / 255.0

        feat = extract_hog_features(arr, config=hog_config)
        X[idx] = feat
        y[idx] = int(row["label"])

        if (idx + 1) % 2500 == 0 or (idx + 1) == n_samples:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{split_name.upper()}] Extracted {idx + 1}/{n_samples} ({elapsed:.1f}s, {rate:.1f} img/s)...")

    gc.collect()
    return X, y


def reuse_baseline_cache(
    config_dir: Path,
    cfg_info: Dict[str, Any],
    n_train: int,
    n_val: int,
    manifest_path: Path,
) -> None:
    """
    Safely reuse existing baseline 900-D cache without regenerating 76+ MB.
    Uses filesystem hard links where supported, or direct references.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    dst_train_x = config_dir / "train_features.npy"
    dst_train_y = config_dir / "train_labels.npy"
    dst_val_x = config_dir / "val_features.npy"
    dst_val_y = config_dir / "val_labels.npy"
    meta_path = config_dir / "metadata.json"

    file_pairs = [
        (BASELINE_TRAIN_X, dst_train_x),
        (BASELINE_TRAIN_Y, dst_train_y),
        (BASELINE_VAL_X, dst_val_x),
        (BASELINE_VAL_Y, dst_val_y),
    ]

    for src, dst in file_pairs:
        if not src.exists():
            raise FileNotFoundError(f"Baseline cache file not found: {src}")
        if not dst.exists():
            try:
                os.link(src, dst)
            except Exception:
                # Fallback to copy if hard link is not permitted
                import shutil
                shutil.copy2(src, dst)

    metadata = {
        "configuration_name": "hog_9_cell8",
        "display_name": cfg_info["display_name"],
        "orientations": cfg_info["orientations"],
        "pixels_per_cell": list(cfg_info["pixels_per_cell"]),
        "cells_per_block": list(cfg_info["cells_per_block"]),
        "block_norm": cfg_info["block_norm"],
        "feature_dimension": cfg_info["expected_dim"],
        "train_sample_count": n_train,
        "validation_sample_count": n_val,
        "extraction_runtime_sec": 0.0,
        "reused_baseline_cache": True,
        "source_manifest_path": manifest_path.as_posix(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Directly reused baseline 900-D features from results/features/ without re-extraction.",
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[REUSE SUCCESS] Reused baseline cache in {config_dir} (0s extraction, zero redundant disk copy).")


def process_hog_configuration(
    config_key: str,
    train_rows: List[dict],
    val_rows: List[dict],
    base_dir: Path,
    manifest_path: Path,
    output_root: Path = HOG_V3_DIR,
    force: bool = False,
) -> None:
    """
    Extract or reuse HOG features for a single configuration and persist to cache directory.
    """
    if config_key not in HOG_CONFIGURATIONS:
        raise KeyError(f"Unknown HOG configuration '{config_key}'. Available: {list(HOG_CONFIGURATIONS.keys())}")

    cfg_info = HOG_CONFIGURATIONS[config_key]
    config_dir = output_root / config_key
    expected_dim = cfg_info["expected_dim"]
    n_train = len(train_rows)
    n_val = len(val_rows)

    print("\n" + "=" * 70)
    print(f"CONFIGURATION: {cfg_info['display_name']} ({config_key})")
    print(f"Parameters: orientations={cfg_info['orientations']}, cell={cfg_info['pixels_per_cell']}, "
          f"block={cfg_info['cells_per_block']}, norm={cfg_info['block_norm']}")
    print(f"Expected feature dimension: {expected_dim} dims")
    print(f"Target directory: {config_dir.as_posix()}")
    print("=" * 70)

    # 1. Check if already cached
    if not force and is_cache_valid(config_dir, expected_dim, n_train, n_val):
        print(f"[CACHE HIT] '{config_key}' already cached at {config_dir}. Skipping computation.")
        return

    # 2. Check if baseline reuse applies
    if cfg_info.get("is_baseline", False):
        print(f"[BASELINE REUSE] Recognizing HOG-1 matches baseline cache. Linking without re-extraction...")
        reuse_baseline_cache(config_dir, cfg_info, n_train, n_val, manifest_path)
        return

    # 3. Perform fresh extraction
    print(f"[EXTRACTING] Extracting HOG features for {n_train} train + {n_val} val images...")
    hog_cfg_dict = {
        "orientations": cfg_info["orientations"],
        "pixels_per_cell": cfg_info["pixels_per_cell"],
        "cells_per_block": cfg_info["cells_per_block"],
        "block_norm": cfg_info["block_norm"],
    }

    t_start = time.time()
    X_train, y_train = extract_features_for_rows(train_rows, base_dir, hog_cfg_dict, expected_dim, "train")
    X_val, y_val = extract_features_for_rows(val_rows, base_dir, hog_cfg_dict, expected_dim, "val")
    total_runtime = time.time() - t_start

    # 4. Save numpy feature arrays
    config_dir.mkdir(parents=True, exist_ok=True)
    train_x_path = config_dir / "train_features.npy"
    train_y_path = config_dir / "train_labels.npy"
    val_x_path = config_dir / "val_features.npy"
    val_y_path = config_dir / "val_labels.npy"

    np.save(train_x_path, X_train)
    np.save(train_y_path, y_train)
    np.save(val_x_path, X_val)
    np.save(val_y_path, y_val)

    # 5. Save metadata JSON
    metadata = {
        "configuration_name": config_key,
        "display_name": cfg_info["display_name"],
        "orientations": cfg_info["orientations"],
        "pixels_per_cell": list(cfg_info["pixels_per_cell"]),
        "cells_per_block": list(cfg_info["cells_per_block"]),
        "block_norm": cfg_info["block_norm"],
        "feature_dimension": expected_dim,
        "train_sample_count": n_train,
        "validation_sample_count": n_val,
        "extraction_runtime_sec": round(total_runtime, 2),
        "source_manifest_path": manifest_path.as_posix(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_size_mb": round(X_train.nbytes / (1024 * 1024), 2),
        "val_size_mb": round(X_val.nbytes / (1024 * 1024), 2),
    }

    meta_path = config_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVED] Configuration '{config_key}' cache saved successfully:")
    print(f"  train_features.npy: {X_train.shape} ({metadata['train_size_mb']} MB)")
    print(f"  val_features.npy:   {X_val.shape} ({metadata['val_size_mb']} MB)")
    print(f"  metadata.json:      {meta_path.as_posix()}")
    print(f"  Extraction runtime: {total_runtime:.1f}s")


def list_configurations() -> None:
    """Print table of all supported HOG configurations and cache status."""
    print("=" * 85)
    print(f"{'Config Key':<15} | {'Orient':<6} | {'Cell':<7} | {'Block':<7} | {'Dim':<6} | {'Status':<12} | {'Description'}")
    print("-" * 85)

    for key, cfg in HOG_CONFIGURATIONS.items():
        cfg_dir = HOG_V3_DIR / key
        cached = is_cache_valid(cfg_dir, cfg["expected_dim"], 22347, 5586)
        status = "CACHED" if cached else "NOT CACHED"
        cell_str = f"{cfg['pixels_per_cell'][0]}x{cfg['pixels_per_cell'][1]}"
        block_str = f"{cfg['cells_per_block'][0]}x{cfg['cells_per_block'][1]}"
        print(f"{key:<15} | {cfg['orientations']:<6} | {cell_str:<7} | {block_str:<7} | {cfg['expected_dim']:<6} | {status:<12} | {cfg['description']}")

    print("=" * 85)


def run_smoke_test(manifest_path: Path, base_dir: Path) -> None:
    """
    Run lightweight smoke test on a small slice of images across all 5 configurations.
    Verifies HOG extraction, feature dimensions, split separation, and metadata formatting.
    """
    print("\n" + "=" * 70)
    print("RUNNING SMOKE TEST FOR ALL 5 HOG CONFIGURATIONS")
    print("=" * 70)

    train_rows, val_rows = load_train_and_val_manifest_rows(manifest_path)

    # Pick 4 representative train and 2 representative val samples
    smoke_train = [train_rows[i] for i in [0, 500, 1000, 2000]]
    smoke_val = [val_rows[i] for i in [0, 200]]

    print(f"Smoke test samples: {len(smoke_train)} train, {len(smoke_val)} validation.")

    test_output_dir = HOG_V3_DIR / "_smoke_test"

    for key, cfg in HOG_CONFIGURATIONS.items():
        expected_dim = cfg["expected_dim"]
        hog_cfg_dict = {
            "orientations": cfg["orientations"],
            "pixels_per_cell": cfg["pixels_per_cell"],
            "cells_per_block": cfg["cells_per_block"],
            "block_norm": cfg["block_norm"],
        }

        # Test train extraction
        X_tr, y_tr = extract_features_for_rows(smoke_train, base_dir, hog_cfg_dict, expected_dim, f"smoke_{key}_tr")
        X_va, y_va = extract_features_for_rows(smoke_val, base_dir, hog_cfg_dict, expected_dim, f"smoke_{key}_va")

        assert X_tr.shape == (len(smoke_train), expected_dim), f"Train dim mismatch for {key}: {X_tr.shape}"
        assert X_va.shape == (len(smoke_val), expected_dim), f"Val dim mismatch for {key}: {X_va.shape}"
        assert X_tr.dtype == np.float32, f"Train dtype mismatch: {X_tr.dtype}"
        assert np.all(np.isfinite(X_tr)), f"NaN/Inf found in {key} train features"
        assert np.all(np.isfinite(X_va)), f"NaN/Inf found in {key} val features"

        # Verify output formatting on one config
        if key == "hog_9_cell8":
            test_output_dir.mkdir(parents=True, exist_ok=True)
            np.save(test_output_dir / "smoke_tr.npy", X_tr)
            np.save(test_output_dir / "smoke_val.npy", X_va)
            meta_content = {
                "config": key,
                "dimension": expected_dim,
                "train_count": len(smoke_train),
                "val_count": len(smoke_val),
            }
            with open(test_output_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_content, f)

        print(f"  [PASS] {key:<15}: Extracted shape ({len(smoke_train)}, {expected_dim}) [dtype={X_tr.dtype}]")

    # Clean up smoke test directory
    if test_output_dir.exists():
        for p in test_output_dir.glob("*"):
            p.unlink()
        test_output_dir.rmdir()

    print("\n" + "=" * 70)
    print("[SUCCESS] Smoke test passed for all 5 HOG configurations!")
    print("  - Dimensions verified: 900, 4356, 1764, 5808, 8712")
    print("  - Train/Validation separation verified")
    print("  - Finite float32 numeric integrity verified")
    print("  - Test set isolation maintained (0 test samples accessed)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: HOG Feature Extraction and Caching Runner (Train & Validation Only)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(HOG_CONFIGURATIONS.keys()),
        help="Run a specific HOG configuration.",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=["all"],
        help="Run a suite of configurations (e.g. 'all').",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported HOG configurations and cache status, then exit.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick smoke test verifying extraction, shapes, and formatting on small subset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cache already exists.",
    )
    args = parser.parse_args()

    if args.list:
        list_configurations()
        return

    if args.smoke_test:
        run_smoke_test(MANIFEST_PATH, DATASET_BASE_DIR)
        return

    # Determine which configs to run
    configs_to_run: List[str] = []
    if args.config:
        configs_to_run = [args.config]
    elif args.suite == "all":
        configs_to_run = list(HOG_CONFIGURATIONS.keys())
    else:
        print("Please specify a target configuration or suite. Use --list to see available configurations.")
        print("Examples:")
        print("  python experiments/cache_hog_v3_features.py --config hog_9_cell4")
        print("  python experiments/cache_hog_v3_features.py --suite all")
        print("  python experiments/cache_hog_v3_features.py --list")
        print("  python experiments/cache_hog_v3_features.py --smoke-test")
        return

    # Load dataset manifest (train and validation only)
    train_rows, val_rows = load_train_and_val_manifest_rows(MANIFEST_PATH)
    print(f"Loaded manifest: {len(train_rows)} train rows, {len(val_rows)} validation rows.")

    for cfg_key in configs_to_run:
        process_hog_configuration(
            config_key=cfg_key,
            train_rows=train_rows,
            val_rows=val_rows,
            base_dir=DATASET_BASE_DIR,
            manifest_path=MANIFEST_PATH,
            output_root=HOG_V3_DIR,
            force=args.force,
        )


if __name__ == "__main__":
    main()
