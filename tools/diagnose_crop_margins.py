import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import mediapipe as mp

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hog_svm import HOGLinearSVM, CLASS_NAMES
from src.features.hog_features import extract_hog_features
from src.features.face_detection import get_face_detector

def test_crop_margins(image_path: str):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return

    model_path = PROJECT_ROOT / "results" / "models" / "hog_svm_final_c035_balanced.joblib"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print("Loading model...")
    model = HOGLinearSVM.load(model_path)
    
    print("Loading image...")
    pil_image = Image.open(image_path)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
        
    width, height = pil_image.size
    
    print("Running Face Detector...")
    tflite_path = str(PROJECT_ROOT / "blaze_face_short_range.tflite")
    try:
        detector = get_face_detector(tflite_path)
    except Exception as e:
        print(f"Error loading face detector: {e}")
        return
        
    np_image = np.array(pil_image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.detections:
        print("No face detected in the image.")
        return
        
    largest_detection = max(
        detection_result.detections,
        key=lambda d: d.bounding_box.width * d.bounding_box.height
    )
    bbox = largest_detection.bounding_box
    x_min, y_min = bbox.origin_x, bbox.origin_y
    box_w, box_h = bbox.width, bbox.height

    margins = [0.00, 0.10, 0.15, 0.20]
    
    print("\nEvaluating margins. Note: LinearSVC decision scores are NOT probabilities.\n")
    print(f"{'MARGIN':<8} | {'CROP_SIZE':<11} | {'PREDICTION':<10} | {'TOP_SCORE':<10} | {'HAPPY_SCORE':<11} | {'SURPRISE_SCORE'}")
    print("-" * 75)
    
    results = []

    for margin in margins:
        margin_x = box_w * margin
        margin_y = box_h * margin
        
        x1 = max(0, int(x_min - margin_x))
        y1 = max(0, int(y_min - margin_y))
        x2 = min(width, int(x_min + box_w + margin_x))
        y2 = min(height, int(y_min + box_h + margin_y))
        
        crop_size = f"{x2-x1}x{y2-y1}"
        cropped = pil_image.crop((x1, y1, x2, y2))
        
        # Exact preprocessing sequence
        img_gray = cropped.convert("L")
        img_resized = img_gray.resize((48, 48), Image.Resampling.BILINEAR)
        img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        
        # Extract HOG
        features = extract_hog_features(img_arr)
        if features.shape[0] != 900:
            print(f"Error: HOG features have wrong dimension {features.shape[0]}")
            return
            
        feature_vec = features.reshape(1, -1)
        
        # Prediction
        decision_scores = model.classifier.decision_function(feature_vec)[0]
        pred_idx = int(model.classifier.predict(feature_vec)[0])
        pred_class = CLASS_NAMES[pred_idx]
        top_score = decision_scores[pred_idx]
        
        happy_idx = CLASS_NAMES.index("happy")
        surprise_idx = CLASS_NAMES.index("surprise")
        happy_score = decision_scores[happy_idx]
        surprise_score = decision_scores[surprise_idx]
        
        print(f"{margin:<8.2f} | {crop_size:<11} | {pred_class:<10} | {top_score:<10.4f} | {happy_score:<11.4f} | {surprise_score:<10.4f}")
        
        results.append((margin, decision_scores))
        
    print("\nDetailed Decision Scores per Margin:")
    print("-" * 50)
    for margin, scores in results:
        print(f"Margin {margin:.2f}:")
        for cls, score in zip(CLASS_NAMES, scores):
            print(f"  {cls:<8}: {score:.4f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the test image")
    args = parser.parse_args()
    test_crop_margins(args.image_path)
