"""
Flask Web Application for Facial Expression Recognition.

Inference and demonstration application for the FINAL HOG + Linear SVM model:
- Model Checkpoint: results/models/hog_svm_final_c035_balanced.joblib
- HOG Configuration: 9 orientations, (8, 8) cells, (2, 2) blocks, L2-Hys norm (900 dimensions)
- Classifier: LinearSVC (C=0.35, class_weight='balanced', random_state=42, max_iter=2000, dual='auto')
- Target Emotions: angry, disgust, fear, happy, neutral, sad, surprise

PREPROCESSING PIPELINE (MediaPipe + Pillow - NO OpenCV):
User uploads photo
  -> Validate file format & integrity
  -> Pillow Image.open()
  -> MediaPipe FaceDetector finds largest face
  -> Crop with 15% margin
  -> Convert to grayscale ('L')
  -> Resize to exactly 48x48
  -> Normalize to float32 in [0.0, 1.0]
  -> Extract 900-D HOG features using authoritative project extractor
  -> LinearSVC decision_function & predict
  -> Display predicted emotion & SVM decision scores

PRIVACY & DATA ISOLATION:
- Processes uploaded images entirely in memory.
- Does NOT permanently store uploaded photographs.
- Does NOT modify any dataset or retrain any model.
- Strictly an inference demo application.
"""

import base64
import io
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.hog_features import HOG_CONFIG, extract_hog_features
from src.features.face_detection import preprocess_face_image
from src.models.hog_svm import CLASS_NAMES


# Canonical Model Location
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "hog_svm_final_c035_balanced.joblib"

# Supported Upload Extensions & Limits
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB limit

# Emotion visual metadata mapping
EMOTION_META: Dict[str, Dict[str, str]] = {
    "angry": {"emoji": "😠", "color": "#f87171", "label": "Angry"},
    "disgust": {"emoji": "🤢", "color": "#34d399", "label": "Disgust"},
    "fear": {"emoji": "😨", "color": "#a78bfa", "label": "Fear"},
    "happy": {"emoji": "😊", "color": "#fbbf24", "label": "Happy"},
    "neutral": {"emoji": "😐", "color": "#94a3b8", "label": "Neutral"},
    "sad": {"emoji": "😢", "color": "#60a5fa", "label": "Sad"},
    "surprise": {"emoji": "😲", "color": "#f472b6", "label": "Surprise"},
}


def load_model_pipeline(model_path: Path = MODEL_PATH):
    """
    Load the final saved HOG + Linear SVM model once at server startup.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Final model checkpoint not found at: {model_path.as_posix()}.\n"
            "Please ensure results/models/hog_svm_final_c035_balanced.joblib exists."
        )

    payload = joblib.load(model_path)
    if isinstance(payload, dict):
        classifier = payload["classifier"]
        hog_cfg = payload.get("hog_config", HOG_CONFIG)
        classes = payload.get("class_names", CLASS_NAMES)
    else:
        classifier = payload
        hog_cfg = HOG_CONFIG
        classes = CLASS_NAMES

    return classifier, hog_cfg, classes


# Global pre-loaded model instances
try:
    CLASSIFIER, ACTIVE_HOG_CONFIG, TARGET_CLASSES = load_model_pipeline(MODEL_PATH)
    print(f"[MODEL LOADED] Successfully loaded final model: {MODEL_PATH.name}")
    print(f"  Classifier: LinearSVC(C={getattr(CLASSIFIER, 'C', 0.35)}, class_weight={getattr(CLASSIFIER, 'class_weight', 'balanced')})")
    print(f"  Target Classes ({len(TARGET_CLASSES)}): {TARGET_CLASSES}")
except Exception as e:
    print(f"[WARNING] Model load deferral: {e}")
    CLASSIFIER = None
    ACTIVE_HOG_CONFIG = HOG_CONFIG
    TARGET_CLASSES = CLASS_NAMES


def is_allowed_file(filename: str) -> bool:
    """Validate file extension against allowed image types."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image_with_pillow(
    image_bytes: bytes,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str], Optional[str]]:
    """
    Preprocessing pipeline using MediaPipe Face Detection + Pillow:
    1. Opens image from in-memory bytes.
    2. Validates image integrity.
    3. Detects and crops largest face (with margin).
    4. Converts to grayscale and resizes to 48x48.
    5. Normalizes to float32 in [0.0, 1.0].
    6. Generates base64 data URLs for both the uploaded image and 48x48 input crop.

    Returns:
        (img_48x48_float32, error_message, uploaded_b64, crop_b64)
    """
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        # Determine format for encoding
        img_format = pil_img.format if pil_img.format else "JPEG"
        if img_format.upper() not in ["PNG", "JPEG", "JPG", "WEBP"]:
            img_format = "JPEG"

        # 1. Convert to RGB for original preview
        preview_rgb = pil_img.convert("RGB")
        buffered = io.BytesIO()
        preview_rgb.save(buffered, format="JPEG", quality=92)
        uploaded_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

        # 2. Detect, crop, convert to grayscale ('L'), and resize to 48x48
        model_asset_path = str(PROJECT_ROOT / "blaze_face_short_range.tflite")
        img_resized = preprocess_face_image(pil_img, model_asset_path=model_asset_path)

        # 3. Normalize to float32 [0.0, 1.0] matching predict_image.py
        img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0

        # 4. Base64 data URL for the 48x48 preprocessed image
        crop_buff = io.BytesIO()
        img_resized.save(crop_buff, format="PNG")
        crop_b64 = "data:image/png;base64," + base64.b64encode(crop_buff.getvalue()).decode("utf-8")

        return img_arr, None, uploaded_b64, crop_b64

    except ValueError as ve:
        # Catch specific face detection errors (e.g. no face detected)
        return None, str(ve), None, None
    except OSError as e:
        # Pillow throws OSError (or UnidentifiedImageError) when it genuinely cannot decode the image
        return None, "Corrupted or invalid image file. Please upload a valid PNG, JPG, or WEBP photo.", None, None
    # Do NOT catch broad exceptions here. If MediaPipe throws RuntimeError (e.g., missing model asset),
    # let it surface so the developer can actually see the missing asset error in the terminal instead
    # of assuming the image stream was corrupted!


def predict_emotion(face_48: np.ndarray) -> Dict[str, Any]:
    """
    Extract 900 HOG features and predict emotion using the final LinearSVC model.
    """
    if CLASSIFIER is None:
        raise RuntimeError("Final model is not loaded. Please verify the model joblib file exists.")

    # 1. Extract 900 HOG features using the authoritative extractor
    features = extract_hog_features(face_48, config=ACTIVE_HOG_CONFIG)
    if features.shape[0] != 900:
        raise ValueError(f"Extracted HOG dimension mismatch: expected 900, got {features.shape[0]}")

    feature_vec = features.reshape(1, -1)

    # 2. Compute hyperplane decision scores & class prediction
    decision_scores = CLASSIFIER.decision_function(feature_vec)[0]
    pred_idx = int(CLASSIFIER.predict(feature_vec)[0])
    predicted_class = TARGET_CLASSES[pred_idx]

    # 3. Softmax-normalized relative scores for visual bars (strictly labeled as 'Relative score', NOT probability)
    exp_scores = np.exp(decision_scores - np.max(decision_scores))
    relative_pcts = (exp_scores / np.sum(exp_scores)) * 100.0

    scores_list = []
    for i, class_name in enumerate(TARGET_CLASSES):
        meta = EMOTION_META.get(class_name, {"emoji": "🎭", "color": "#818cf8", "label": class_name.capitalize()})
        scores_list.append({
            "class_name": class_name,
            "label": meta["label"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "raw_score": round(float(decision_scores[i]), 3),
            "relative_score": round(float(relative_pcts[i]), 1),
            "is_winner": (i == pred_idx),
        })

    # Sort descending by raw SVM decision score
    scores_list.sort(key=lambda s: s["raw_score"], reverse=True)

    winner_meta = EMOTION_META.get(predicted_class, {"emoji": "😊", "label": predicted_class.capitalize()})

    return {
        "predicted_class": predicted_class,
        "predicted_label": winner_meta["label"],
        "predicted_emoji": winner_meta["emoji"],
        "scores": scores_list,
    }


def create_app() -> Flask:
    """Create and configure Flask application instance."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            result=None,
            error=None,
            model_loaded=(CLASSIFIER is not None),
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if CLASSIFIER is None:
            return render_template(
                "index.html",
                result=None,
                error="Model is not available. Please ensure results/models/hog_svm_final_c035_balanced.joblib exists.",
                model_loaded=False,
            )

        # 1. Check for file in request
        if "image" not in request.files:
            return render_template("index.html", result=None, error="No image uploaded. Please choose an image file.")

        file = request.files["image"]
        if file.filename == "":
            return render_template("index.html", result=None, error="No image selected. Please browse and select a photo.")

        if not is_allowed_file(file.filename):
            return render_template(
                "index.html",
                result=None,
                error=f"Unsupported file format. Please upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
            )

        # 2. Read file in memory (ZERO permanent disk storage)
        try:
            image_bytes = file.read()
            if len(image_bytes) == 0:
                return render_template("index.html", result=None, error="Empty file uploaded. Please upload a valid image.")
        except Exception:
            return render_template("index.html", result=None, error="Failed to read uploaded file.")

        # 3. Preprocess image with Pillow (no face detector)
        t_start = time.time()
        face_48, error_msg, uploaded_b64, crop_b64 = preprocess_image_with_pillow(image_bytes)

        if face_48 is None:
            return render_template("index.html", result=None, error=error_msg, model_loaded=True)

        # 4. Predict emotion using HOG + Linear SVM
        try:
            prediction_output = predict_emotion(face_48)
            inference_time_ms = round((time.time() - t_start) * 1000, 1)

            result_data = {
                "prediction": prediction_output,
                "uploaded_image": uploaded_b64,
                "crop_image": crop_b64,
                "latency_ms": inference_time_ms,
            }

            # Return JSON for AJAX requests or HTML template for standard form post
            if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.accept_mimetypes.best == "application/json":
                return jsonify({"success": True, "data": result_data})

            return render_template("index.html", result=result_data, error=None, model_loaded=True)

        except Exception as e:
            return render_template(
                "index.html",
                result=None,
                error=f"Prediction error: {str(e)}",
                model_loaded=True,
            )

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "model_loaded": (CLASSIFIER is not None),
            "model_type": "HOG (900-D) + LinearSVC (C=0.35, balanced)",
            "preprocessing": "MediaPipe (Face Crop) + Pillow (Grayscale -> 48x48 -> float32 [0,1])",
            "classes": TARGET_CLASSES,
        })

    return app


# Application entrypoint
app = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("STARTING FACIAL EXPRESSION RECOGNITION FLASK SERVER")
    print("=" * 65)
    print("Model:         HOG (900-D) + Linear Support Vector Machine (C=0.35, balanced)")
    print("Features:      9 orientations, 8x8 cell, 2x2 block, L2-Hys")
    print("Trained On:    FER2013 (Train + Validation = 27,933 samples)")
    print("Preprocessing: MediaPipe (Face Crop) + Pillow (Grayscale -> 48x48) (NO OpenCV)")
    print("URL:           http://127.0.0.1:5000")
    print("=" * 65 + "\n")

    app.run(host="127.0.0.1", port=5000, debug=False)
