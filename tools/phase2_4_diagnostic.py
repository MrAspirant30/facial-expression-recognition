"""
Phase 2: FER2013 Production Round-Trip Diagnostic.

Compares validation accuracy between:
  PATH A: Native FER2013 48x48 -> HOG -> final model (ground truth pipeline)
  PATH B: FER2013 48x48 -> upsample ~400x400 -> MediaPipe face detect -> crop -> 48x48 -> HOG -> final model

Also computes:
  - Overall accuracy and Macro F1 for both paths
  - Per-class recall (especially HAPPY and SURPRISE)
  - Prediction agreement A vs B
  - MediaPipe detection failure rate
  - OOD score analysis (max decision score distributions)

Phase 3: Perturbation robustness on validation subset and phone image.

Phase 4: OOD / score diagnostic.

All using validation data only. Test set is NEVER touched.
"""

import csv
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import HOG_CONFIG, extract_hog_features
from src.models.hog_svm import HOGLinearSVM, CLASS_NAMES

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MANIFEST = PROJECT_ROOT / "data" / "fer2013" / "archive" / "clean_dataset_split.csv"
DATASET_BASE = MANIFEST.parent
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "hog_svm_final_c035_balanced.joblib"
TFLITE_PATH = str(PROJECT_ROOT / "blaze_face_short_range.tflite")
DIAG_DIR = PROJECT_ROOT / "results" / "diagnostics"

# Use a stratified subsample for tractability
SAMPLES_PER_CLASS = 100  # 700 total from validation
RNG_SEED = 42


def load_validation_rows():
    """Load validation rows from manifest."""
    rows = []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "validation":
                rows.append(row)
    return rows


def stratified_sample(rows, n_per_class, seed=42):
    """Return a stratified random subsample of validation rows."""
    rng = np.random.RandomState(seed)
    by_class = {}
    for row in rows:
        label = int(row["label"])
        by_class.setdefault(label, []).append(row)
    
    sampled = []
    for label in sorted(by_class.keys()):
        pool = by_class[label]
        k = min(n_per_class, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        for idx in chosen:
            sampled.append(pool[idx])
    return sampled


def path_a_inference(row, model):
    """Native FER2013 pipeline: open 48x48 grayscale -> float32/255 -> HOG -> predict."""
    img_path = DATASET_BASE / row["path"]
    img = Image.open(img_path)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    feat = extract_hog_features(arr, config=HOG_CONFIG)
    fv = feat.reshape(1, -1)
    scores = model.classifier.decision_function(fv)[0]
    pred = int(model.classifier.predict(fv)[0])
    return pred, scores


def path_b_inference(row, model, detector):
    """
    Production round-trip: 
    FER 48x48 -> upsample to ~400x400 -> MediaPipe detect -> crop with 15% margin
    -> grayscale -> 48x48 BILINEAR -> /255 -> HOG -> predict.
    
    Returns (pred, scores, detected) where detected is False if MediaPipe failed.
    """
    import mediapipe as mp
    
    img_path = DATASET_BASE / row["path"]
    img = Image.open(img_path)
    
    # Upsample to ~400x400 to simulate real-world resolution
    upscale_size = (400, 400)
    img_up = img.convert("RGB").resize(upscale_size, Image.Resampling.BILINEAR)
    
    np_img = np.array(img_up)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
    
    result = detector.detect(mp_img)
    
    if not result.detections:
        return None, None, False
    
    det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bbox = det.bounding_box
    w, h = img_up.size
    margin = 0.15
    mx = bbox.width * margin
    my = bbox.height * margin
    x1 = max(0, int(bbox.origin_x - mx))
    y1 = max(0, int(bbox.origin_y - my))
    x2 = min(w, int(bbox.origin_x + bbox.width + mx))
    y2 = min(h, int(bbox.origin_y + bbox.height + my))
    
    crop = img_up.crop((x1, y1, x2, y2))
    gray = crop.convert("L")
    resized = gray.resize((48, 48), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    
    feat = extract_hog_features(arr, config=HOG_CONFIG)
    fv = feat.reshape(1, -1)
    scores = model.classifier.decision_function(fv)[0]
    pred = int(model.classifier.predict(fv)[0])
    return pred, scores, True


def run_perturbation_analysis(image_arr_48, model, label_name="unknown"):
    """
    Phase 3: Test prediction stability under small perturbations.
    Returns dict of results.
    """
    base_feat = extract_hog_features(image_arr_48, config=HOG_CONFIG)
    base_pred = int(model.classifier.predict(base_feat.reshape(1, -1))[0])
    
    perturbation_results = {"base_pred": base_pred, "variants": []}
    
    h, w = image_arr_48.shape
    
    # Translation perturbations
    for dx, dy in [(2, 0), (-2, 0), (0, 2), (0, -2), (4, 0), (-4, 0), (0, 4), (0, -4)]:
        shifted = np.zeros_like(image_arr_48)
        src_x1 = max(0, dx)
        src_y1 = max(0, dy)
        src_x2 = min(w, w + dx)
        src_y2 = min(h, h + dy)
        dst_x1 = max(0, -dx)
        dst_y1 = max(0, -dy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = image_arr_48[src_y1:src_y2, src_x1:src_x2]
        
        feat = extract_hog_features(shifted, config=HOG_CONFIG)
        pred = int(model.classifier.predict(feat.reshape(1, -1))[0])
        perturbation_results["variants"].append({
            "type": f"translate({dx},{dy})",
            "pred": pred,
            "same": pred == base_pred,
        })
    
    # Horizontal flip
    flipped = np.fliplr(image_arr_48)
    feat = extract_hog_features(flipped, config=HOG_CONFIG)
    pred = int(model.classifier.predict(feat.reshape(1, -1))[0])
    perturbation_results["variants"].append({
        "type": "hflip",
        "pred": pred,
        "same": pred == base_pred,
    })
    
    # Small rotations via Pillow
    pil_img = Image.fromarray((image_arr_48 * 255).astype(np.uint8), mode="L")
    for angle in [5, -5, 10, -10]:
        rotated = pil_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        rot_arr = np.asarray(rotated, dtype=np.float32) / 255.0
        feat = extract_hog_features(rot_arr, config=HOG_CONFIG)
        pred = int(model.classifier.predict(feat.reshape(1, -1))[0])
        perturbation_results["variants"].append({
            "type": f"rotate({angle})",
            "pred": pred,
            "same": pred == base_pred,
        })
    
    # Scale changes (crop center, resize back)
    for scale in [0.9, 1.1]:
        new_sz = int(48 * scale)
        if scale < 1.0:
            # Zoom in: crop center
            offset = (48 - new_sz) // 2
            cropped = image_arr_48[offset:offset+new_sz, offset:offset+new_sz]
            pil_c = Image.fromarray((cropped * 255).astype(np.uint8), mode="L")
            scaled = pil_c.resize((48, 48), Image.Resampling.BILINEAR)
        else:
            # Zoom out: pad with zeros
            pil_c = Image.fromarray((image_arr_48 * 255).astype(np.uint8), mode="L")
            pil_c = pil_c.resize((new_sz, new_sz), Image.Resampling.BILINEAR)
            offset = (new_sz - 48) // 2
            cropped = np.asarray(pil_c)[offset:offset+48, offset:offset+48]
            scaled = Image.fromarray(cropped, mode="L")
        
        s_arr = np.asarray(scaled, dtype=np.float32) / 255.0
        feat = extract_hog_features(s_arr, config=HOG_CONFIG)
        pred = int(model.classifier.predict(feat.reshape(1, -1))[0])
        perturbation_results["variants"].append({
            "type": f"scale({scale})",
            "pred": pred,
            "same": pred == base_pred,
        })
    
    return perturbation_results


def main():
    import mediapipe as mp
    from src.features.face_detection import get_face_detector
    
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("COMPREHENSIVE FER2013 DIAGNOSTIC — PHASES 2, 3, 4")
    print("=" * 70)
    print("Using VALIDATION data only. Test set is LOCKED.\n")
    
    # Load model
    model = HOGLinearSVM.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH.name}")
    print(f"  C={model.C}, HOG dim={extract_hog_features(np.zeros((48,48), dtype=np.float32)).shape[0]}")
    
    # Load detector
    detector = get_face_detector(TFLITE_PATH)
    print("MediaPipe face detector loaded.\n")
    
    # Load validation rows and stratified sample
    all_val = load_validation_rows()
    sample = stratified_sample(all_val, SAMPLES_PER_CLASS, RNG_SEED)
    print(f"Validation total: {len(all_val)}, Stratified sample: {len(sample)}")
    label_counts = Counter(int(r["label"]) for r in sample)
    for label in sorted(label_counts):
        print(f"  Class {label} ({CLASS_NAMES[label]}): {label_counts[label]}")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: Round-trip comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: PRODUCTION ROUND-TRIP TEST")
    print("=" * 70)
    
    y_true = []
    preds_a = []
    preds_b = []
    scores_a_list = []
    scores_b_list = []
    mp_failures = 0
    
    t_start = time.time()
    for i, row in enumerate(sample):
        label = int(row["label"])
        y_true.append(label)
        
        # Path A
        pred_a, sc_a = path_a_inference(row, model)
        preds_a.append(pred_a)
        scores_a_list.append(sc_a)
        
        # Path B
        pred_b, sc_b, detected = path_b_inference(row, model, detector)
        if detected:
            preds_b.append(pred_b)
            scores_b_list.append(sc_b)
        else:
            mp_failures += 1
            preds_b.append(-1)  # sentinel
            scores_b_list.append(None)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(sample)} ({time.time()-t_start:.1f}s)")
    
    elapsed = time.time() - t_start
    print(f"  Completed in {elapsed:.1f}s\n")
    
    y_true = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    
    # Path A metrics
    acc_a = accuracy_score(y_true, preds_a)
    f1_a = f1_score(y_true, preds_a, average="macro", zero_division=0)
    recall_a = recall_score(y_true, preds_a, average=None, zero_division=0)
    
    print("PATH A (Native FER2013 pipeline):")
    print(f"  Accuracy:  {acc_a:.4f}")
    print(f"  Macro F1:  {f1_a:.4f}")
    print(f"  Per-class recall:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:<10}: {recall_a[i]:.4f}")
    
    # Path B metrics (excluding MediaPipe failures)
    valid_mask = preds_b >= 0
    n_valid = valid_mask.sum()
    
    print(f"\nPATH B (Production round-trip pipeline):")
    print(f"  MediaPipe detection failures: {mp_failures}/{len(sample)} ({100*mp_failures/len(sample):.1f}%)")
    
    if n_valid > 0:
        acc_b = accuracy_score(y_true[valid_mask], preds_b[valid_mask])
        f1_b = f1_score(y_true[valid_mask], preds_b[valid_mask], average="macro", zero_division=0)
        recall_b = recall_score(y_true[valid_mask], preds_b[valid_mask], average=None, zero_division=0)
        
        print(f"  Accuracy:  {acc_b:.4f} (on {n_valid} detected)")
        print(f"  Macro F1:  {f1_b:.4f}")
        print(f"  Per-class recall:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name:<10}: {recall_b[i]:.4f}")
    
    # Agreement
    agree_mask = valid_mask & (preds_a == preds_b)
    agreement_rate = agree_mask.sum() / n_valid if n_valid > 0 else 0
    print(f"\n  A-B prediction agreement: {agree_mask.sum()}/{n_valid} ({100*agreement_rate:.1f}%)")
    
    # Disagreement analysis
    disagree_mask = valid_mask & (preds_a != preds_b)
    if disagree_mask.sum() > 0:
        print(f"\n  Disagreements ({disagree_mask.sum()} samples):")
        switch_counts = Counter()
        for idx in np.where(disagree_mask)[0]:
            switch_counts[(CLASS_NAMES[preds_a[idx]], CLASS_NAMES[preds_b[idx]])] += 1
        for (a_cls, b_cls), count in switch_counts.most_common(15):
            print(f"    A={a_cls:<10} -> B={b_cls:<10}: {count}")
    
    # Confusion matrices
    print("\n  Path A Confusion Matrix:")
    cm_a = confusion_matrix(y_true, preds_a, labels=range(7))
    header = "         " + " ".join(f"{n[:4]:>5}" for n in CLASS_NAMES)
    print(f"  {header}")
    for i, name in enumerate(CLASS_NAMES):
        row_str = " ".join(f"{cm_a[i,j]:5d}" for j in range(7))
        print(f"  {name[:4]:>8} {row_str}")
    
    if n_valid > 0:
        print("\n  Path B Confusion Matrix (detected only):")
        cm_b = confusion_matrix(y_true[valid_mask], preds_b[valid_mask], labels=range(7))
        print(f"  {header}")
        for i, name in enumerate(CLASS_NAMES):
            row_str = " ".join(f"{cm_b[i,j]:5d}" for j in range(7))
            print(f"  {name[:4]:>8} {row_str}")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 3: Perturbation robustness
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: PERTURBATION ROBUSTNESS")
    print("=" * 70)
    
    # Test on 50 random validation samples
    rng = np.random.RandomState(99)
    perturb_idxs = rng.choice(len(sample), size=min(50, len(sample)), replace=False)
    
    val_stability = []
    for idx in perturb_idxs:
        row = sample[idx]
        img = Image.open(DATASET_BASE / row["path"])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        res = run_perturbation_analysis(arr, model, CLASS_NAMES[int(row["label"])])
        n_same = sum(1 for v in res["variants"] if v["same"])
        val_stability.append(n_same / len(res["variants"]))
    
    print(f"\nFER2013 Validation samples (n={len(perturb_idxs)}):")
    print(f"  Mean prediction stability: {np.mean(val_stability):.4f}")
    print(f"  Median: {np.median(val_stability):.4f}")
    print(f"  Min: {np.min(val_stability):.4f}, Max: {np.max(val_stability):.4f}")
    
    # Test on phone image
    phone_path = Path(r"C:\Users\jyoti\OneDrive\Desktop\MOBILE\Photos_only\1736416243_IMG-20250109-WA0011.jpg")
    if phone_path.exists():
        phone_img = Image.open(phone_path)
        rgb = phone_img.convert("RGB")
        np_rgb = np.array(rgb)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_rgb)
        det_result = detector.detect(mp_img)
        
        if det_result.detections:
            det = max(det_result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bbox = det.bounding_box
            pw, ph = phone_img.size
            margin = 0.15
            mx = bbox.width * margin
            my = bbox.height * margin
            x1 = max(0, int(bbox.origin_x - mx))
            y1 = max(0, int(bbox.origin_y - my))
            x2 = min(pw, int(bbox.origin_x + bbox.width + mx))
            y2 = min(ph, int(bbox.origin_y + bbox.height + my))
            crop = phone_img.crop((x1, y1, x2, y2))
            gray = crop.convert("L")
            resized = gray.resize((48, 48), Image.Resampling.BILINEAR)
            phone_arr = np.asarray(resized, dtype=np.float32) / 255.0
            
            phone_perturb = run_perturbation_analysis(phone_arr, model, "phone_happy")
            n_same = sum(1 for v in phone_perturb["variants"] if v["same"])
            stability = n_same / len(phone_perturb["variants"])
            
            print(f"\nPhone image perturbation stability: {stability:.4f}")
            print(f"  Base prediction: {CLASS_NAMES[phone_perturb['base_pred']]}")
            for v in phone_perturb["variants"]:
                status = "SAME" if v["same"] else f"CHANGED -> {CLASS_NAMES[v['pred']]}"
                print(f"    {v['type']:<20}: {status}")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 4: OOD / Score Diagnostic
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: SCORE / OOD DIAGNOSTIC")
    print("=" * 70)
    
    # Collect max decision scores and margins for Path A (validation)
    max_scores_a = []
    margins_a = []
    for sc in scores_a_list:
        sorted_sc = np.sort(sc)[::-1]
        max_scores_a.append(sorted_sc[0])
        margins_a.append(sorted_sc[0] - sorted_sc[1])
    
    max_scores_a = np.array(max_scores_a)
    margins_a = np.array(margins_a)
    
    print("\nValidation Path A score distribution:")
    print(f"  Max decision score: mean={max_scores_a.mean():.4f}, std={max_scores_a.std():.4f}")
    print(f"    Percentiles: 5th={np.percentile(max_scores_a, 5):.4f}, "
          f"25th={np.percentile(max_scores_a, 25):.4f}, "
          f"50th={np.percentile(max_scores_a, 50):.4f}, "
          f"75th={np.percentile(max_scores_a, 75):.4f}, "
          f"95th={np.percentile(max_scores_a, 95):.4f}")
    print(f"  Top-1 vs Top-2 margin: mean={margins_a.mean():.4f}, std={margins_a.std():.4f}")
    
    # Phone image scores
    if phone_path.exists() and det_result.detections:
        phone_feat = extract_hog_features(phone_arr, config=HOG_CONFIG)
        phone_fv = phone_feat.reshape(1, -1)
        phone_scores = model.classifier.decision_function(phone_fv)[0]
        phone_max = np.max(phone_scores)
        phone_sorted = np.sort(phone_scores)[::-1]
        phone_margin = phone_sorted[0] - phone_sorted[1]
        phone_pred = CLASS_NAMES[int(model.classifier.predict(phone_fv)[0])]
        
        # Compute percentile rank of phone score in validation distribution
        pct_rank = np.mean(max_scores_a <= phone_max) * 100
        
        print(f"\nPhone image score analysis:")
        print(f"  Prediction: {phone_pred}")
        print(f"  Max decision score: {phone_max:.4f}")
        print(f"  Top-1 vs Top-2 margin: {phone_margin:.4f}")
        print(f"  Percentile rank in validation: {pct_rank:.1f}th")
        print(f"  All scores: {dict(zip(CLASS_NAMES, [f'{s:.4f}' for s in phone_scores]))}")
        
        # Is all-negative unusual?
        all_neg_val = np.sum(np.all(np.array(scores_a_list) < 0, axis=1))
        print(f"\n  Validation samples with ALL decision scores < 0: {all_neg_val}/{len(scores_a_list)} ({100*all_neg_val/len(scores_a_list):.1f}%)")
        print(f"  Phone image has all scores < 0: {np.all(phone_scores < 0)}")
    
    # HOG feature-space distance analysis
    print("\nHOG feature-space distance analysis:")
    
    # Compute class centroids from validation Path A features
    val_features_a = []
    for row in sample:
        img = Image.open(DATASET_BASE / row["path"])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        feat = extract_hog_features(arr, config=HOG_CONFIG)
        val_features_a.append(feat)
    val_features_a = np.array(val_features_a)
    
    centroids = {}
    for label in range(7):
        mask = y_true == label
        centroids[label] = val_features_a[mask].mean(axis=0)
    
    # Phone image distance to each centroid
    if phone_path.exists() and det_result.detections:
        print(f"\n  Phone image L2 distance to class centroids:")
        for label in range(7):
            dist = np.linalg.norm(phone_feat - centroids[label])
            print(f"    {CLASS_NAMES[label]:<10}: {dist:.4f}")
        
        # Compare with typical validation distances
        print(f"\n  Typical validation L2 distances to own-class centroid:")
        for label in range(7):
            mask = y_true == label
            class_feats = val_features_a[mask]
            dists = np.linalg.norm(class_feats - centroids[label], axis=1)
            print(f"    {CLASS_NAMES[label]:<10}: mean={dists.mean():.4f}, std={dists.std():.4f}")
    
    # ═══════════════════════════════════════════════════════════
    # Abstention threshold experiment
    # ═══════════════════════════════════════════════════════════
    print("\n" + "-" * 70)
    print("ABSTENTION THRESHOLD EXPERIMENT (validation only)")
    print("-" * 70)
    
    # Find threshold on max decision score that gives ~90% correct among non-abstained
    thresholds = np.arange(-1.0, 1.5, 0.1)
    print(f"\n{'Threshold':<12} {'Abstained':<12} {'Remaining':<12} {'Accuracy':<12} {'Coverage':<12}")
    print("-" * 60)
    for thr in thresholds:
        accepted = max_scores_a >= thr
        n_accepted = accepted.sum()
        if n_accepted == 0:
            continue
        acc = accuracy_score(y_true[accepted], preds_a[accepted])
        coverage = n_accepted / len(y_true)
        print(f"{thr:<12.2f} {(~accepted).sum():<12d} {n_accepted:<12d} {acc:<12.4f} {coverage:<12.4f}")
    
    # Would phone image be abstained?
    if phone_path.exists() and det_result.detections:
        print(f"\n  Phone image max score ({phone_max:.4f}) would be abstained at thresholds > {phone_max:.2f}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nIMPORTANT: LinearSVC decision scores are NOT probabilities.")
    print("All analysis uses validation data only. Test set was NOT touched.")


if __name__ == "__main__":
    main()
