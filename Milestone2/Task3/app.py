"""
Milestone 2 : task3
=====================
Task 3: Identify and Extract Key Information from OCR Text

OCR will return raw text, but we need to extract structured details like:
Essential Fields to Extract:
● Payee Name (&quot;Pay to the Order of&quot;)
● Amount (Numerical and Written)
● Date
● Check Number
● Bank Name
● Account Number &amp; Routing Number (MICR line at the bottom)

Methods for Structured Extraction:
● Use Regular Expressions (Regex) to detect specific patterns in text.
● Implement Named Entity Recognition (NER) using NLP libraries (SpaCy, BERT) for
better accuracy.
● If OCR output is messy, use Rule-Based Filtering to remove unwanted text.
"""


import os
import cv2
import re
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Configure Tesseract Path (if not in your system PATH)
# -------------------------------------------------------
# On Windows, Tesseract might be installed at:
#    C:\Program Files\Tesseract-OCR\tesseract.exe
# Adjust the path below if needed:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -------------------------------------------------------
# 2. Load the Image
# -------------------------------------------------------
image_path = r'C:\Users\NAVNATH\OneDrive\Desktop\Infosys Internship\Automating-Bank-Check-Extraction-from-Scanned-PDFs_Feb_2025\Milestone2\Task1\Extracted_Checks\page1\page1_check1.jpg'  # <<-- Update this path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError("Could not read the image. Check file integrity and path.")

# -------------------------------------------------------
# 3. Preprocess the Image (Grayscale, Blur, Threshold)
# -------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological operation (closing) to remove noise/gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Median blur to reduce any remaining salt-and-pepper noise
processed = cv2.medianBlur(morphed, 3)

# -------------------------------------------------------
# 4. (Optional) Display the Preprocessed Image
# -------------------------------------------------------
plt.imshow(processed, cmap='gray')
plt.title("Preprocessed Check")
plt.axis('off')
plt.show()

# -------------------------------------------------------
# 5. Perform OCR with Tesseract
# -------------------------------------------------------
extracted_text = pytesseract.image_to_string(processed)
print("===== RAW OCR TEXT =====")
print(extracted_text)
print("========================")

# -------------------------------------------------------
# 6. Basic Parsing of Fields (Heuristic / Regex Approach)
# -------------------------------------------------------
# Note: Real-world checks may need more advanced or template-based parsing.

lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]

# Initialize possible fields
payee_name = None
date = None
amount_words = None
amount_numeric = None
check_number = None

# Simple regex patterns (tweak as needed)
date_pattern = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b')
# Looks for typical numeric patterns like 12,20,000 or 1220000
amount_pattern = re.compile(r'(\d{1,3}(,\d{2,3})*(\.\d{1,2})?)')
check_number_pattern = re.compile(r'(ch(e|i)que\s*no\.?|check\s*no\.?|cheque\s*#|check\s*#)\s*(\w+)', re.IGNORECASE)

for line in lines:
    # Try to find a date
    if not date:
        match_date = date_pattern.search(line)
        if match_date:
            date = match_date.group(1)

    # Try to find a numeric amount
    # This is naive; it might capture partial amounts. Tweak to your check format.
    match_amount = amount_pattern.search(line)
    if match_amount:
        # Heuristic: If it looks big enough to be a check amount, store it
        # (or you can store all and decide later).
        candidate = match_amount.group(1)
        # For example, if we find "12,20,000" or "12000" or "12,000"
        # We'll just store the first one we see as the 'amount_numeric'
        if not amount_numeric:
            amount_numeric = candidate

    # Look for typical text that might indicate payee name
    # In some checks, the payee's name might appear near "Pay to the order of" or "OR BEARER"
    # but that is not always standard. We'll do a naive guess: if line doesn't have digits or
    # typical words like "Rupees", "Date", "A/C", it might be the payee name.
    # This is extremely naive.
    if (not payee_name and
        'rupees' not in line.lower() and
        'account' not in line.lower() and
        'a/c' not in line.lower() and
        not any(char.isdigit() for char in line) and
        len(line.split()) <= 4 and  # heuristic: payee names are often short
        len(line.split()) >= 2):
        payee_name = line

    # Try to detect check number from known keywords
    match_check_no = check_number_pattern.search(line)
    if match_check_no:
        check_number = match_check_no.group(2)

# Attempt to extract amount in words if there's a line containing "rupees" or "only" etc.
for line in lines:
    if ('rupees' in line.lower() or 'only' in line.lower()) and len(line.split()) > 2:
        amount_words = line
        break

# -------------------------------------------------------
# 7. Print Parsed Fields
# -------------------------------------------------------
print("\n===== PARSED FIELDS =====")
print(f"Payee Name:    {payee_name}")
print(f"Date:          {date}")
print(f"Amount (Words): {amount_words}")
print(f"Amount (Numeric): {amount_numeric}")
print(f"Check Number:  {check_number}")
print("=========================")
