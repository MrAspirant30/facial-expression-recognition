import mediapipe as mp
import numpy as np
from PIL import Image

# Global configuration for Face Crop Margin
FACE_CROP_MARGIN = 0.15

# Global detector instance (lazy load)
_face_detector = None

def get_face_detector(model_asset_path: str):
    """Lazy load the MediaPipe face detector."""
    global _face_detector
    if _face_detector is None:
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_asset_path),
            running_mode=VisionRunningMode.IMAGE
        )
        _face_detector = FaceDetector.create_from_options(options)
    return _face_detector

def detect_and_crop_face(pil_image: Image.Image, model_asset_path: str = "blaze_face_short_range.tflite") -> Image.Image:
    """
    Detect faces using MediaPipe, select the largest one, 
    and crop it with FACE_CROP_MARGIN.
    Raises ValueError if no face is detected.
    """
    detector = get_face_detector(model_asset_path)
    
    # MediaPipe requires RGB format for inference
    if pil_image.mode != "RGB":
        rgb_image = pil_image.convert("RGB")
    else:
        rgb_image = pil_image
        
    np_image = np.array(rgb_image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_image)
    
    # Perform detection
    detection_result = detector.detect(mp_image)
    
    if not detection_result.detections:
        raise ValueError("Could not detect a face. Please upload a clearer, front-facing photo.")
        
    # Select the largest face by bounding box area
    largest_detection = max(
        detection_result.detections, 
        key=lambda d: d.bounding_box.width * d.bounding_box.height
    )
    
    bbox = largest_detection.bounding_box
    width, height = pil_image.size
    
    # Calculate margins
    x_min = bbox.origin_x
    y_min = bbox.origin_y
    box_w = bbox.width
    box_h = bbox.height
    
    margin_x = box_w * FACE_CROP_MARGIN
    margin_y = box_h * FACE_CROP_MARGIN
    
    # Clip coordinates to image boundaries
    x1 = max(0, int(x_min - margin_x))
    y1 = max(0, int(y_min - margin_y))
    x2 = min(width, int(x_min + box_w + margin_x))
    y2 = min(height, int(y_min + box_h + margin_y))
    
    # Crop from the original image to preserve color space until final preprocessing
    return pil_image.crop((x1, y1, x2, y2))

def preprocess_face_image(pil_image: Image.Image, model_asset_path: str = "blaze_face_short_range.tflite") -> Image.Image:
    """
    Standard pre-processing pipeline for the HOG SVM model:
    1. Detect and crop largest face with margin.
    2. Convert to grayscale ('L').
    3. Resize to exact 48x48.
    """
    # 1. Detect and crop face
    cropped_face = detect_and_crop_face(pil_image, model_asset_path)
    
    # 2. Convert to grayscale and resize to 48x48
    img_gray = cropped_face.convert("L")
    img_resized = img_gray.resize((48, 48), Image.Resampling.BILINEAR)
    
    return img_resized
