"""
Milestone 1 : task2
=====================
1.Develop an algorithm for detecting and isolating individual checks
from scanned PDF images.

2. Use image processing libraries such as OpenCV to identify check
boundaries and crop them.

"""
import streamlit as st
import cv2
import numpy as np
import os
from pdf2image import convert_from_bytes

def extract_checks_from_image(image):
    """
    Process the input image and return a list of extracted check images.
    """
    if image is None:
        return []
    
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Adaptive thresholding to isolate potential check regions
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Morphological operations to clean up the image
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and aspect ratio for typical check dimensions
    min_area = 50000  # adjust as needed
    max_area = 300000
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:
                valid_contours.append(cnt)
    
    # If no valid checks are found, use the largest contour as a fallback
    if not valid_contours:
        valid_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    
    extracted_checks = []
    for cnt in valid_contours:
        # Get the rotated rectangle and compute the perspective transform
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        width = int(rect[1][0])
        height = int(rect[1][1])
        if width == 0 or height == 0:
            continue  # Avoid invalid transforms
        
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(orig, M, (width, height))
        extracted_checks.append(warped)
    
    return extracted_checks

# ---------------- Streamlit GUI ----------------

st.set_page_config(page_title="PDF Check Extractor", layout="wide")
st.title("PDF Check Extractor")
st.markdown("""
Upload a PDF file. Each page will be converted into an image and processed to extract check regions.
Extracted checks will be saved automatically in an **Extracted_Checks** folder with subfolders per page.
""")

# PDF file uploader widget
uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
rotate_original = st.checkbox("Rotate original page image horizontally for display", value=False)

if uploaded_pdf is not None:
    # Convert PDF pages to images
    pages = convert_from_bytes(uploaded_pdf.read())
    st.write(f"Found {len(pages)} page(s) in the PDF.")
    
    # Create the base directory for saving extracted checks
    base_folder = "Extracted_Checks"
    os.makedirs(base_folder, exist_ok=True)
    
    # Process each page from the PDF
    for i, page in enumerate(pages):
        st.header(f"Page {i+1}")
        # Convert the PIL image to a NumPy array and then to an OpenCV BGR image
        page_np = np.array(page)
        image_cv = cv2.cvtColor(page_np, cv2.COLOR_RGB2BGR)
        
        # Optionally flip the displayed original image horizontally
        if rotate_original:
            display_image = cv2.flip(page_np, 1)
        else:
            display_image = page_np
        
        st.image(display_image, caption=f"Original Page {i+1}", use_column_width=True)
        
        # Extract checks from the unrotated original image
        extracted_checks = extract_checks_from_image(image_cv)
        
        if extracted_checks:
            st.success(f"Extracted {len(extracted_checks)} check(s) from page {i+1}!")
            # Create a subfolder for the current page
            page_folder = os.path.join(base_folder, f"page{i+1}")
            os.makedirs(page_folder, exist_ok=True)
            
            for j, check_img in enumerate(extracted_checks):
                # Build file name and file path
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
        else:
            st.error(f"No check found in page {i+1}.")
