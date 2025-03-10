
"""
1.Implement PDF to image conversion functionality using libraries like
PyPDF2 .

2. Split multipage PDFs into individual pages and convert them into
images.
"""
import os
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image

poppler_path = r"C:\poppler-24.08.0\Library\bin"
# Function to convert PDF to images
def pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Converts each page of a PDF into an image and saves them in the output folder.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_folder (str): Path to the folder where images will be saved.
        dpi (int): Dots per inch for the output images (default: 300).

    Returns:
        list: List of file paths for the saved images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)

    # Save each page as an image
    saved_image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        saved_image_paths.append(image_path)

    return saved_image_paths

# Streamlit App
def main():
    st.title("Bank Check Extraction - PDF to Image Converter")
    st.write("Upload a scanned PDF file to extract individual pages as images.")

    # File upload button
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_pdf_path = os.path.join("temp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Submit button
        if st.button("Convert PDF to Images"):
            # Create a temporary folder for images
            output_folder = "temp_images"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Show progress bar
            st.write("Converting PDF to images...")
            progress_bar = st.progress(0)

            # Convert PDF to images
            saved_images = pdf_to_images(temp_pdf_path, output_folder, dpi=300)

            # Update progress bar
            progress_bar.progress(100)
            st.success("Conversion completed!")

            # Display extracted images
            st.write("### Extracted Images")
            for i, image_path in enumerate(saved_images):
                st.write(f"**Page {i + 1}**")
                image = Image.open(image_path)
                st.image(image, caption=f"Page {i + 1}", use_column_width=True)

            # Clean up temporary files
            os.remove(temp_pdf_path)

# Run the Streamlit app
if __name__ == "__main__":
    # Create a temporary folder for uploaded files
    if not os.path.exists("temp"):
        os.makedirs("temp")
    if not os.path.exists("temp_images"):
        os.makedirs("temp_images")

    main()