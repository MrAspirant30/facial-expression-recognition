"""
Clean Validation Experiment for FER2013 HOG+SVM.
1. Trains a baseline model using ONLY the Train split (no validation leakage).
2. Evaluates the pristine Validation split using Path A (native HOG) vs Path B (MediaPipe round-trip).
3. Evaluates an augmented training strategy (5 spatial perturbations per image) to see if it closes the Path A/B gap.
"""

import sys
import time
from pathlib import Path
import csv
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import HOG_CONFIG, extract_hog_features
from src.models.hog_svm import HOGLinearSVM, CLASS_NAMES
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import mediapipe as mp
from src.features.face_detection import get_face_detector

# Setup
MANIFEST = PROJECT_ROOT / "data" / "fer2013" / "archive" / "clean_dataset_split.csv"
DATASET_BASE = MANIFEST.parent
FEATURES_DIR = PROJECT_ROOT / "results" / "features"
TFLITE_PATH = str(PROJECT_ROOT / "blaze_face_short_range.tflite")
RNG_SEED = 42
rng = np.random.RandomState(RNG_SEED)

def get_augmentations(img_arr):
    """Generate 5 augmented versions of a 48x48 float32 image."""
    variants = [img_arr]
    
    # 1. Flip
    variants.append(np.fliplr(img_arr))
    
    pil_img = Image.fromarray((img_arr * 255).astype(np.uint8), mode="L")
    
    # 2. Random Translate (+/- 4px)
    dx, dy = rng.randint(-4, 5, size=2)
    shifted = np.zeros_like(img_arr)
    h, w = img_arr.shape
    src_x1, src_y1 = max(0, dx), max(0, dy)
    src_x2, src_y2 = min(w, w + dx), min(h, h + dy)
    dst_x1, dst_y1 = max(0, -dx), max(0, -dy)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        shifted[dst_y1:dst_y2, dst_x1:dst_x2] = img_arr[src_y1:src_y2, src_x1:src_x2]
    variants.append(shifted)
    
    # 3. Random Rotate (+/- 10 deg)
    angle = rng.uniform(-10, 10)
    rotated = pil_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    variants.append(np.asarray(rotated, dtype=np.float32) / 255.0)
    
    # 4. Random Scale (0.9 to 1.1)
    scale = rng.uniform(0.9, 1.1)
    new_sz = int(48 * scale)
    if scale < 1.0:
        offset = (48 - new_sz) // 2
        cropped = img_arr[offset:offset+new_sz, offset:offset+new_sz]
        scaled = Image.fromarray((cropped * 255).astype(np.uint8), mode="L").resize((48, 48), Image.Resampling.BILINEAR)
    else:
        scaled_pil = pil_img.resize((new_sz, new_sz), Image.Resampling.BILINEAR)
        offset = (new_sz - 48) // 2
        cropped = np.asarray(scaled_pil)[offset:offset+48, offset:offset+48]
        scaled = Image.fromarray(cropped, mode="L")
    variants.append(np.asarray(scaled, dtype=np.float32) / 255.0)
    
    return variants

def evaluate_model(model, val_rows, detector, model_name):
    print(f"\n==================================================")
    print(f"EVALUATING: {model_name}")
    print(f"==================================================")
    
    y_true = []
    preds_a = []
    preds_b = []
    mp_failures = 0
    
    print(f"Running Path A and Path B on {len(val_rows)} validation samples...")
    t0 = time.time()
    
    for i, row in enumerate(val_rows):
        label = int(row["label"])
        y_true.append(label)
        img_path = DATASET_BASE / row["path"]
        
        # PATH A: Native FER2013 Pipeline
        img = Image.open(img_path)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        feat_a = extract_hog_features(arr, config=HOG_CONFIG)
        pred_a = int(model.classifier.predict(feat_a.reshape(1, -1))[0])
        preds_a.append(pred_a)
        
        # PATH B: MediaPipe Round-Trip
        img_up = img.convert("RGB").resize((400, 400), Image.Resampling.BILINEAR)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img_up))
        result = detector.detect(mp_img)
        
        if result.detections:
            det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bbox = det.bounding_box
            w, h = img_up.size
            mx, my = bbox.width * 0.15, bbox.height * 0.15
            x1, y1 = max(0, int(bbox.origin_x - mx)), max(0, int(bbox.origin_y - my))
            x2, y2 = min(w, int(bbox.origin_x + bbox.width + mx)), min(h, int(bbox.origin_y + bbox.height + my))
            crop = img_up.crop((x1, y1, x2, y2))
            resized = crop.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
            arr_b = np.asarray(resized, dtype=np.float32) / 255.0
            feat_b = extract_hog_features(arr_b, config=HOG_CONFIG)
            pred_b = int(model.classifier.predict(feat_b.reshape(1, -1))[0])
            preds_b.append(pred_b)
        else:
            mp_failures += 1
            preds_b.append(-1)
            
        if (i+1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(val_rows)}...")

    print(f"Completed in {time.time()-t0:.1f}s")
    
    y_true = np.array(y_true)
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    
    # Metrics Path A
    acc_a = accuracy_score(y_true, preds_a)
    p_a = precision_score(y_true, preds_a, average="macro", zero_division=0)
    r_a = recall_score(y_true, preds_a, average="macro", zero_division=0)
    f1_a = f1_score(y_true, preds_a, average="macro", zero_division=0)
    recalls_a = recall_score(y_true, preds_a, average=None, zero_division=0)
    
    print("\n--- PATH A (Native Validation) ---")
    print(f"Accuracy:        {acc_a:.4f}")
    print(f"Macro Precision: {p_a:.4f}")
    print(f"Macro Recall:    {r_a:.4f}")
    print(f"Macro F1:        {f1_a:.4f}")
    print(f"Happy Recall:    {recalls_a[3]:.4f}")
    print(f"Surprise Recall: {recalls_a[6]:.4f}")
    
    # Metrics Path B
    valid_mask = preds_b >= 0
    n_valid = valid_mask.sum()
    print(f"\n--- PATH B (MediaPipe Round-Trip) ---")
    print(f"MediaPipe detection failures: {mp_failures}/{len(val_rows)} ({100*mp_failures/len(val_rows):.1f}%)")
    
    if n_valid > 0:
        y_valid = y_true[valid_mask]
        p_valid = preds_b[valid_mask]
        acc_b = accuracy_score(y_valid, p_valid)
        p_b = precision_score(y_valid, p_valid, average="macro", zero_division=0)
        r_b = recall_score(y_valid, p_valid, average="macro", zero_division=0)
        f1_b = f1_score(y_valid, p_valid, average="macro", zero_division=0)
        recalls_b = recall_score(y_valid, p_valid, average=None, zero_division=0)
        
        print(f"Accuracy:        {acc_b:.4f}")
        print(f"Macro Precision: {p_b:.4f}")
        print(f"Macro Recall:    {r_b:.4f}")
        print(f"Macro F1:        {f1_b:.4f}")
        print(f"Happy Recall:    {recalls_b[3]:.4f}")
        print(f"Surprise Recall: {recalls_b[6]:.4f}")
        
        # Agreement
        agree_mask = (preds_a[valid_mask] == p_valid)
        print(f"\nPath A/B Agreement: {agree_mask.sum()}/{n_valid} ({100*agree_mask.sum()/n_valid:.1f}%)")


def main():
    print("=" * 60)
    print("CLEAN DIAGNOSTIC: TRAIN-ONLY MODELS")
    print("=" * 60)
    
    detector = get_face_detector(TFLITE_PATH)
    
    train_rows, val_rows = [], []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train": train_rows.append(r)
            elif r["split"] == "validation": val_rows.append(r)
            
    print(f"FER2013 Dataset Loaded: {len(train_rows)} Train, {len(val_rows)} Validation.")
    
    # ---------------------------------------------------------
    # 1. Baseline Train-Only Model
    # ---------------------------------------------------------
    print("\nLoading cached Train split features...")
    X_train = np.load(FEATURES_DIR / "train_features.npy")
    y_train = np.load(FEATURES_DIR / "train_labels.npy")
    print(f"Shape: X={X_train.shape}, y={y_train.shape}")
    
    print("\nTraining BASELINE Diagnostic Model (LinearSVC C=0.35, dual=False)...")
    from sklearn.svm import LinearSVC
    t0 = time.time()
    # Using dual=False is much faster for n_samples > n_features and mathematically equivalent.
    baseline_svm = LinearSVC(C=0.35, class_weight="balanced", random_state=42, max_iter=2000, dual=False)
    baseline_svm.fit(X_train, y_train)
    print(f"Training completed in {time.time()-t0:.1f}s")
    
    baseline_model = HOGLinearSVM(C=0.35, random_state=42, max_iter=2000, hog_config=HOG_CONFIG)
    baseline_model.classifier = baseline_svm
    
    evaluate_model(baseline_model, val_rows, detector, "BASELINE HOG+SVM (Train Only)")
    
    # ---------------------------------------------------------
    # 2. Augmented Train-Only Model
    # ---------------------------------------------------------
    print("\n\n" + "=" * 60)
    print("TRAINING AUGMENTED DIAGNOSTIC MODEL")
    print("=" * 60)
    print(f"Generating 5 augmented variants per training sample (approx 111k total)...")
    
    X_aug, y_aug = [], []
    t0 = time.time()
    for i, row in enumerate(train_rows):
        img = Image.open(DATASET_BASE / row["path"])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        label = int(row["label"])
        for variant in get_augmentations(arr):
            X_aug.append(extract_hog_features(variant, config=HOG_CONFIG))
            y_aug.append(label)
        if (i+1) % 5000 == 0:
            print(f"  Augmented {i+1}/{len(train_rows)} ({time.time()-t0:.1f}s)")
            
    X_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug, dtype=np.int64)
    print(f"Augmented Train Shape: X={X_aug.shape}, y={y_aug.shape}")
    
    print("\nTraining AUGMENTED Diagnostic Model (LinearSVC C=0.35, dual=False)...")
    t0 = time.time()
    aug_svm = LinearSVC(C=0.35, class_weight="balanced", random_state=42, max_iter=2000, dual=False)
    aug_svm.fit(X_aug, y_aug)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f}s")
    
    aug_model = HOGLinearSVM(C=0.35, random_state=42, max_iter=2000, hog_config=HOG_CONFIG)
    aug_model.classifier = aug_svm
    
    evaluate_model(aug_model, val_rows, detector, "AUGMENTED HOG+SVM (Train Only)")
    
    # ---------------------------------------------------------
    # 3. Check Phone Image against both models
    # ---------------------------------------------------------
    phone_path = Path(r"C:\Users\jyoti\OneDrive\Desktop\MOBILE\Photos_only\1736416243_IMG-20250109-WA0011.jpg")
    if phone_path.exists():
        print("\n\n" + "=" * 60)
        print("REAL PHONE PHOTOGRAPH INFERENCE")
        print("=" * 60)
        img = Image.open(phone_path)
        img_up = img.convert("RGB")
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img_up))
        res = detector.detect(mp_img)
        if res.detections:
            det = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bbox = det.bounding_box
            w, h = img_up.size
            mx, my = bbox.width * 0.15, bbox.height * 0.15
            x1, y1 = max(0, int(bbox.origin_x - mx)), max(0, int(bbox.origin_y - my))
            x2, y2 = min(w, int(bbox.origin_x + bbox.width + mx)), min(h, int(bbox.origin_y + bbox.height + my))
            crop = img_up.crop((x1, y1, x2, y2))
            resized = crop.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            feat = extract_hog_features(arr, config=HOG_CONFIG)
            fv = feat.reshape(1, -1)
            
            p_base = int(baseline_model.classifier.predict(fv)[0])
            p_aug = int(aug_model.classifier.predict(fv)[0])
            
            print(f"Baseline Model Prediction:  {CLASS_NAMES[p_base]}")
            print(f"Augmented Model Prediction: {CLASS_NAMES[p_aug]}")

if __name__ == "__main__":
    main()
