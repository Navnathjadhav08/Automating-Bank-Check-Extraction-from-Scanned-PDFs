import cv2
import numpy as np

def calculate_skew_angle(binary_image):
    """Calculate rotation angle using minAreaRect on binary image features."""
    coords = np.column_stack(np.where(binary_image > 0))
    if len(coords) < 500:  # Minimum features threshold
        return 0.0
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    return angle + 90 if angle < -45 else angle

def correct_skew(image, angle):
    """Rotate image by calculated angle around center with replication border handling."""
    
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

def preprocess_image(image_path, output_path):
    """Enhanced preprocessing pipeline optimized for check OCR."""
    # Load and validate image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image load failed - check file path and integrity")

    # Initial processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
    enhanced = clahe.apply(blurred)

    # Temporary binarization for skew detection
    temp_binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 5
    )
    
    # Skew correction
    angle = calculate_skew_angle(temp_binary)
    deskewed_gray = correct_skew(enhanced, angle)
    
    # Final binarization
    final_thresh = cv2.adaptiveThreshold(
        deskewed_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )

    # Noise removal
    denoised = cv2.morphologyEx(
        final_thresh, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=2
    )

    # Edge-preserving smoothing
    smoothed = cv2.medianBlur(denoised, 3)

    # Save as lossless PNG for OCR
    cv2.imwrite(output_path, smoothed)
    return smoothed

# Usage example (maintain same interface)
input_path = r"Automating-Bank-Check-Extraction-from-Scanned-PDFs_Feb_2025\Milestone2\Task1\Extracted_Checks\page1\page1_check1.jpg"
output_path = "preprocessed_check.png"
preprocessed_image = preprocess_image(input_path, output_path)

print(f"Optimized preprocessed image saved to {output_path}")