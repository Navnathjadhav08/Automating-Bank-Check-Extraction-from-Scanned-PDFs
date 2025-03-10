"""
Milestone 2 : task 2
=====================
Extract Text Using OCR (Tesseract or Other OCR Engine)
"""

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import os

# If Tesseract is not in your PATH, specify its location:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Step 1: Load the image
image_path = r'Automating-Bank-Check-Extraction-from-Scanned-PDFs_Feb_2025\Milestone1\Task3\Extracted_Checks\page2\page2_check1.jpg'  # Replace with your actual image path

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError("Error reading the image file. Check file integrity and path.")

# Step 2: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Remove Noise with Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Apply Thresholding to Convert to Binary Image (using Otsu's thresholding)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 5: Optional Morphological Operations to close gaps or remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Optional: Further noise reduction with median blur
processed = cv2.medianBlur(morphed, 3)

# Step 6: Save and Display the Preprocessed Image
cv2.imwrite('preprocessed_image.jpg', processed)

# Display the image using matplotlib (suitable for most environments)
plt.imshow(processed, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

# Step 7: Extract Text using Tesseract OCR
extracted_text = pytesseract.image_to_string(processed)
print("Extracted Text:")
print(extracted_text)
