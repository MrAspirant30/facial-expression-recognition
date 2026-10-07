"""
Experiment: Train Augmented HOG+SVM.
Trains a HOG+LinearSVC model with spatial data augmentation to combat the
framing brittleness and pipeline mismatch identified in the Phase 2-4 diagnostics.

Methodology:
1. Load FER2013 Train split.
2. For each image, generate 5 variants:
   - Original
   - Horizontal Flip
   - Random Translate (-4 to +4 px)
   - Random Rotate (-10 to +10 deg)
   - Random Scale (0.9 to 1.1)
3. Extract HOG features.
4. Train LinearSVC.
5. Evaluate on Validation set (Native Path A).
6. Evaluate on Validation set (Round-trip Path B with MediaPipe).
"""

import csv
import sys
import time
from pathlib import Path
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

MANIFEST = PROJECT_ROOT / "data" / "fer2013" / "archive" / "clean_dataset_split.csv"
DATASET_BASE = MANIFEST.parent
TFLITE_PATH = str(PROJECT_ROOT / "blaze_face_short_range.tflite")
RNG_SEED = 42
rng = np.random.RandomState(RNG_SEED)

def get_augmentations(img_arr):
    """Generate 5 augmented versions of a 48x48 float32 image."""
    variants = [img_arr]
    
    # 1. Flip
    variants.append(np.fliplr(img_arr))
    
    pil_img = Image.fromarray((img_arr * 255).astype(np.uint8), mode="L")
    
    # 2. Random Translate
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
    
    # 3. Random Rotate
    angle = rng.uniform(-10, 10)
    rotated = pil_img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    variants.append(np.asarray(rotated, dtype=np.float32) / 255.0)
    
    # 4. Random Scale
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

def path_b_inference(row, model, detector):
    img_path = DATASET_BASE / row["path"]
    img = Image.open(img_path)
    img_up = img.convert("RGB").resize((400, 400), Image.Resampling.BILINEAR)
    np_img = np.array(img_up)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
    result = detector.detect(mp_img)
    
    if not result.detections:
        return None, False
        
    det = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bbox = det.bounding_box
    w, h = img_up.size
    margin = 0.15
    mx, my = bbox.width * margin, bbox.height * margin
    x1, y1 = max(0, int(bbox.origin_x - mx)), max(0, int(bbox.origin_y - my))
    x2, y2 = min(w, int(bbox.origin_x + bbox.width + mx)), min(h, int(bbox.origin_y + bbox.height + my))
    
    crop = img_up.crop((x1, y1, x2, y2))
    resized = crop.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    feat = extract_hog_features(arr, config=HOG_CONFIG)
    pred = int(model.classifier.predict(feat.reshape(1, -1))[0])
    return pred, True

def main():
    print("Loading train and validation rows...")
    train_rows, val_rows = [], []
    with open(MANIFEST, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "train": train_rows.append(r)
            elif r["split"] == "validation": val_rows.append(r)
            
    print(f"Generating augmented training data from {len(train_rows)} samples...")
    X_train, y_train = [], []
    t0 = time.time()
    for i, row in enumerate(train_rows):
        img = Image.open(DATASET_BASE / row["path"])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        label = int(row["label"])
        for variant in get_augmentations(arr):
            X_train.append(extract_hog_features(variant, config=HOG_CONFIG))
            y_train.append(label)
        if (i+1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(train_rows)} ({time.time()-t0:.1f}s)")
            
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    print(f"Augmented Train Shape: {X_train.shape}")
    
    print("\nExtracting Validation Path A features...")
    X_val = []
    y_val = []
    for row in val_rows:
        img = Image.open(DATASET_BASE / row["path"])
        arr = np.asarray(img, dtype=np.float32) / 255.0
        X_val.append(extract_hog_features(arr, config=HOG_CONFIG))
        y_val.append(int(row["label"]))
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int64)
    
    print("\nTraining LinearSVC(C=0.35, balanced) on Augmented Data...")
    t0 = time.time()
    model = HOGLinearSVM(C=0.35, random_state=42, max_iter=2000, hog_config=HOG_CONFIG)
    model.fit(X_train, y_train)
    print(f"Training completed in {time.time()-t0:.1f}s")
    
    # Path A Eval
    preds_a = model.predict(X_val)
    acc_a = accuracy_score(y_val, preds_a)
    f1_a = f1_score(y_val, preds_a, average="macro")
    print(f"\nValidation PATH A (Native FER): Accuracy={acc_a:.4f}, Macro F1={f1_a:.4f}")
    
    # Path B Eval
    print("\nEvaluating Validation PATH B (MediaPipe Round-Trip)...")
    detector = get_face_detector(TFLITE_PATH)
    preds_b, y_true_b = [], []
    
    # Subsample validation for speed in Path B (e.g. 1000 samples)
    rng2 = np.random.RandomState(99)
    sample_idxs = rng2.choice(len(val_rows), size=min(1000, len(val_rows)), replace=False)
    
    for i, idx in enumerate(sample_idxs):
        row = val_rows[idx]
        pred, det = path_b_inference(row, model, detector)
        if det:
            preds_b.append(pred)
            y_true_b.append(int(row["label"]))
    
    acc_b = accuracy_score(y_true_b, preds_b)
    f1_b = f1_score(y_true_b, preds_b, average="macro")
    print(f"Validation PATH B: Accuracy={acc_b:.4f}, Macro F1={f1_b:.4f}")
    
    # Let's also predict on the phone image
    phone_path = Path(r"C:\Users\jyoti\OneDrive\Desktop\MOBILE\Photos_only\1736416243_IMG-20250109-WA0011.jpg")
    if phone_path.exists():
        img = Image.open(phone_path)
        img_up = img.convert("RGB")
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img_up))
        res = detector.detect(mp_img)
        if res.detections:
            det = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            bbox = det.bounding_box
            w, h = img_up.size
            margin = 0.15
            mx, my = bbox.width * margin, bbox.height * margin
            x1, y1 = max(0, int(bbox.origin_x - mx)), max(0, int(bbox.origin_y - my))
            x2, y2 = min(w, int(bbox.origin_x + bbox.width + mx)), min(h, int(bbox.origin_y + bbox.height + my))
            crop = img_up.crop((x1, y1, x2, y2))
            resized = crop.convert("L").resize((48, 48), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            feat = extract_hog_features(arr, config=HOG_CONFIG)
            fv = feat.reshape(1, -1)
            pred = int(model.classifier.predict(fv)[0])
            scores = model.classifier.decision_function(fv)[0]
            print(f"\nPhone Image Augmented Prediction: {CLASS_NAMES[pred]}")
            print(f"Scores: {dict(zip(CLASS_NAMES, [f'{s:.4f}' for s in scores]))}")

if __name__ == "__main__":
    main()
