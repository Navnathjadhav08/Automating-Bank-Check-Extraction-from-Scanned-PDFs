"""
Milestone 2 : task4
=====================
Task 4: Validate and Store Extracted Data
Once key details are extracted, we need to ensure the data is correct and ready for further
processing.

✅ Steps:
● Validate the amount format (e.g., does &quot;$1,250.00&quot; match the written text?).
● Ensure the check number matches the MICR line.
● Standardize extracted data into JSON format for database storage.
"""


#############################################################################
# *****************Finalize code **********
#############################################################################
import streamlit as st
import cv2
import numpy as np
import os
import re
from pdf2image import convert_from_bytes
import pytesseract
import tempfile
from datetime import datetime
from PIL import Image
import psycopg2
import psycopg2.extras
import pandas as pd

# ----------------------------------------------------------------------
# Configure Tesseract (Adjust path if needed, or comment out if in PATH)
# ----------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------------------------------------------------------------
# PostgreSQL Database Functions
# ----------------------------------------------------------------------
def get_connection(connection_info):
    """
    Returns a psycopg2 connection using the provided dictionary.
    Example connection_info:
        {
            "host": "localhost",
            "dbname": "mydatabase",
            "user": "myuser",
            "password": "mypassword",
            "port": 5432
        }
    """
    return psycopg2.connect(**connection_info)

def create_table_if_not_exists(connection_info):
    """
    Creates the 'checks' table if it doesn't exist.
    """
    create_sql = """
    CREATE TABLE IF NOT EXISTS checks (
        id SERIAL PRIMARY KEY,
        payee_name      VARCHAR(100),
        check_date      VARCHAR(50),
        amount_words    VARCHAR(255),
        amount_numeric  VARCHAR(50),
        check_number    VARCHAR(50),
        bank_name       VARCHAR(100),
        micr_line       VARCHAR(255),
        raw_ocr_text    TEXT,
        inserted_at     TIMESTAMP DEFAULT NOW()
    );
    """
    conn = get_connection(connection_info)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
    finally:
        conn.close()

def insert_check_data(connection_info, data):
    """
    Insert a single check record into the 'checks' table.
    'data' is a dict with keys: payee_name, check_date, amount_words,
    amount_numeric, check_number, bank_name, micr_line, raw_ocr_text.
    """
    insert_sql = """
        INSERT INTO checks (
            payee_name, check_date, amount_words, amount_numeric,
            check_number, bank_name, micr_line, raw_ocr_text
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    conn = get_connection(connection_info)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (
                    data.get("payee_name"),
                    data.get("check_date"),
                    data.get("amount_words"),
                    data.get("amount_numeric"),
                    data.get("check_number"),
                    data.get("bank_name"),
                    data.get("micr_line"),
                    data.get("raw_ocr_text")
                ))
    finally:
        conn.close()

def get_all_checks(connection_info):
    """
    Retrieve all check records from the 'checks' table as a Pandas DataFrame.
    """
    select_sql = "SELECT * FROM checks ORDER BY id DESC;"
    conn = get_connection(connection_info)
    try:
        with conn:
            df = pd.read_sql(select_sql, conn)
    finally:
        conn.close()
    return df

# ----------------------------------------------------------------------
# Utility: Order contour points for perspective transform
# ----------------------------------------------------------------------
def order_points(pts):
    # Initialize a list of coordinates in the order: bottom-left, top-left, top-right, bottom-right
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0])]
    
    # Split into left and right points
    left = x_sorted[:2]
    right = x_sorted[2:]
    
    # Sort left points by y-coordinate to get top-left and bottom-left
    left = left[np.argsort(left[:, 1])]
    (top_left, bottom_left) = left
    
    # Sort right points by y-coordinate to get top-right and bottom-right
    right = right[np.argsort(right[:, 1])]
    (top_right, bottom_right) = right
    
    return np.array([bottom_left, top_left, top_right, bottom_right], dtype="float32")

# ----------------------------------------------------------------------
# Extract checks from a page image
# ----------------------------------------------------------------------
def extract_checks_from_image(image):
    """
    Given a page image (OpenCV BGR), find possible check regions and warp them out.
    Returns a list of extracted check images.
    """
    if image is None:
        return []
    
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adaptive threshold (inverted)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Morphological closing
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and aspect ratio
    min_area = 50000
    max_area = 300000
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:  # typical check shape
                valid_contours.append(cnt)
    
    # If no valid contour found, just take the largest one
    if not valid_contours:
        valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    
    extracted_checks = []
    for cnt in valid_contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        ordered_box = order_points(box)

        # Calculate real dimensions
        (bottom_left, top_left, top_right, _) = ordered_box
        width = int(np.linalg.norm(top_right - top_left))
        height = int(np.linalg.norm(top_left - bottom_left))
        width = max(width, 1)
        height = max(height, 1)

        # Destination points
        dst_pts = np.array([
            [0, height-1],         
            [0, 0],                
            [width-1, 0],         
            [width-1, height-1]    
        ], dtype="float32")

        # Perspective transform
        M = cv2.getPerspectiveTransform(ordered_box, dst_pts)
        warped = cv2.warpPerspective(orig, M, (width, height))
        
        extracted_checks.append(warped)
    
    return extracted_checks

# ----------------------------------------------------------------------
# OCR + Field Extraction
# ----------------------------------------------------------------------
def ocr_and_parse_check(check_image):
    """
    Perform OCR on the extracted check image, then parse:
      - Payee Name
      - Amount (numeric & words)
      - Date
      - Check Number
      - Bank Name (naive approach)
      - MICR line (account/routing)
    Returns a dictionary of extracted fields.
    """
    # Convert to grayscale, threshold, etc. for better OCR
    gray = cv2.cvtColor(check_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OCR
    text = pytesseract.image_to_string(binarized)
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    payee_name = None
    date_str = None
    amount_words = None
    amount_numeric = None
    check_number = None
    bank_name = None
    micr_line = None

    # Regex patterns
    date_pattern = re.compile(r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b')
    amount_pattern = re.compile(r'(\d{1,3}(,\d{2,3})*(\.\d{1,2})?)')
    check_number_pattern = re.compile(r'(ch(e|i)que\s*no\.?|check\s*no\.?|cheque\s*#|check\s*#)\s*(\w+)', re.IGNORECASE)
    micr_pattern = re.compile(r'[0-9]{5,}')  # naive: sequence of 5+ digits
    bank_keywords = ["bank", "axis", "barclays", "wells fargo", "chase", "hdfc", "sbi", "icici"]

    # Heuristic parsing
    for line in lines:
        # Date
        if not date_str:
            m_date = date_pattern.search(line)
            if m_date:
                date_str = m_date.group(1)

        # Amount numeric
        if not amount_numeric:
            m_amt = amount_pattern.search(line)
            if m_amt:
                amount_numeric = m_amt.group(1)

        # Check number
        if not check_number:
            m_check = check_number_pattern.search(line)
            if m_check:
                check_number = m_check.group(2)

        # Bank name
        if not bank_name:
            lower_line = line.lower()
            for keyword in bank_keywords:
                if keyword in lower_line:
                    bank_name = line
                    break

        # Potential MICR line
        if micr_pattern.search(line):
            if not micr_line:
                micr_line = line

        # Naive payee detection
        if (not payee_name and
            'rupees' not in line.lower() and
            'account' not in line.lower() and
            'a/c' not in line.lower() and
            not any(char.isdigit() for char in line) and
            len(line.split()) <= 4 and
            len(line.split()) >= 2):
            payee_name = line

    # Attempt to extract amount in words
    for line in lines:
        if ('rupees' in line.lower() or 'only' in line.lower()) and len(line.split()) > 2:
            amount_words = line
            break

    return {
        "payee_name": payee_name,
        "check_date": date_str,
        "amount_words": amount_words,
        "amount_numeric": amount_numeric,
        "check_number": check_number,
        "bank_name": bank_name,
        "micr_line": micr_line,
        "raw_ocr_text": text
    }

def validate_data(parsed_fields):
    """
    Basic validations:
      - numeric amount format
      - check number vs. MICR line
    """
    # Validate numeric amount
    amount_str = parsed_fields.get("amount_numeric")
    if amount_str:
        clean_amount = amount_str.replace(",", "")
        try:
            float(clean_amount)  # Just to confirm it's float-like
        except ValueError:
            st.warning(f"Invalid numeric amount format: {amount_str}")
            parsed_fields["amount_numeric"] = None

    # Check number vs. MICR
    check_no = parsed_fields.get("check_number")
    micr = parsed_fields.get("micr_line")
    if check_no and micr and check_no not in micr:
        st.warning(f"Check number '{check_no}' not found in MICR line '{micr}'")

# ----------------------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------------------
st.set_page_config(page_title="PDF Check Extractor + OCR + DB", layout="wide")
st.title("PDF Check Extractor + OCR & Data Validation (PostgreSQL Integration)")

st.markdown("""
**Milestone 1 : Task 3**  
- Saves each detected check as a separate image.

**Milestone 2 : Task 2**  
- Extract text from each check using Tesseract OCR.

**Milestone 2 : Task 3**  
- Identify and extract key fields (payee name, amount, date, check number, bank name, MICR).

**Milestone 2 : Task 4**  
- Validate extracted data and store it in a **PostgreSQL** database.
""")

# Database connection info (adjust for your environment)
connection_info = {
    "host": "localhost",
    "dbname": "mydatabase",
    "user": "myuser",
    "password": "mypassword",
    "port": 5432
}

# Ensure the checks table exists
create_table_if_not_exists(connection_info)

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
rotate_original = st.checkbox("Rotate original page image horizontally for display", value=False)

if uploaded_pdf is not None:
    pages = convert_from_bytes(uploaded_pdf.read())
    st.write(f"Found {len(pages)} page(s) in the PDF.")

    # Create base directory for saving extracted checks
    base_folder = "Extracted_Checks"
    os.makedirs(base_folder, exist_ok=True)
    
    for i, page in enumerate(pages):
        st.header(f"Page {i+1}")

        # Convert PIL page to OpenCV (BGR)
        page_np = np.array(page)
        image_cv = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
        
        # Optionally flip the displayed original image horizontally
        if rotate_original:
            display_image = cv2.flip(page_np, 1)
        else:
            display_image = page_np
        
        st.image(display_image, caption=f"Original Page {i+1}", use_column_width=True)

        # Extract checks
        extracted_checks = extract_checks_from_image(image_cv)
        
        if extracted_checks:
            st.success(f"Extracted {len(extracted_checks)} check(s) from page {i+1}!")
            page_folder = os.path.join(base_folder, f"page{i+1}")
            os.makedirs(page_folder, exist_ok=True)

            for j, check_img in enumerate(extracted_checks):
                # Save each check (Milestone 1: Task 3)
                check_filename = f"page{i+1}_check{j+1}.jpg"
                file_path = os.path.join(page_folder, check_filename)
                cv2.imwrite(file_path, check_img)

                # Display the extracted check
                check_img_rgb = cv2.cvtColor(check_img, cv2.COLOR_BGR2RGB)
                st.image(check_img_rgb, caption=f"Extracted Check {j+1} from Page {i+1}", use_column_width=True)

                # Provide a download button for each extracted check
                is_success, buffer = cv2.imencode(".jpg", check_img)
                if is_success:
                    st.download_button(
                        label=f"Download {check_filename}",
                        data=buffer.tobytes(),
                        file_name=check_filename,
                        mime="image/jpeg",
                        key=f"download_{i+1}_{j+1}"
                    )
                
                # OCR + Field Extraction
                parsed_fields = ocr_and_parse_check(check_img)
                
                # Display parsed fields
                st.write("**Parsed Fields**:")
                st.json(parsed_fields)

                # Validate & Store in DB
                if st.button(f"OCR & Store Check {j+1} from Page {i+1}", key=f"store_{i+1}_{j+1}"):
                    validate_data(parsed_fields)
                    insert_check_data(connection_info, parsed_fields)
                    st.success(f"Check {j+1} data stored successfully!")
        else:
            st.error(f"No check found in page {i+1}.")

st.subheader("View Stored Checks from Database")
if st.button("Refresh Database Records"):
    df_checks = get_all_checks(connection_info)
    if df_checks.empty:
        st.info("No checks found in the database.")
    else:
        st.dataframe(df_checks)
