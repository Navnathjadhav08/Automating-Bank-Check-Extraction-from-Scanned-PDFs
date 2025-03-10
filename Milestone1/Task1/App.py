"""
Milestone 1 : task1
=====================
1.Implement PDF to image conversion functionality using libraries like
PyPDF2 .

2. Split multipage PDFs into individual pages and convert them into
images.
"""
import os
import time
import streamlit as st
from pdf2image import convert_from_path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyPDF2 import PdfReader

# Configuration
DEFAULT_DPI = 200
MAX_THREADS = 4

def create_unique_folder(base_path, pdf_name):
    """Create timestamped folder for each PDF"""
    folder_name = f"{os.path.splitext(pdf_name)[0]}_{int(time.time())}"
    full_path = os.path.join(base_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def get_page_count(pdf_path):
    """Get total number of pages in the PDF"""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        return len(reader.pages)

def convert_page(pdf_path, page_num, output_folder, dpi):
    """Convert a single PDF page to an image"""
    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1)
    img_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
    images[0].save(img_path, "PNG")
    return img_path

def process_pdf_with_progress(pdf_path, output_folder, dpi, progress_bar):
    """Process PDF with real-time progress tracking"""
    total_pages = get_page_count(pdf_path)
    image_paths = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = [executor.submit(convert_page, pdf_path, page_num, output_folder, dpi) 
                  for page_num in range(total_pages)]

        completed = 0
        for future in as_completed(futures):
            image_paths.append(future.result())
            completed += 1
            progress = completed / total_pages
            progress_bar.progress(progress, text=f"Converted {completed}/{total_pages} pages")

    return sorted(image_paths)

def main():
    st.set_page_config(page_title="PDF Check Processor", layout="wide")

    # Initialize session state
    if 'current_image' not in st.session_state:
        st.session_state.update({
            'current_image': 0,
            'processed_pdfs': {},
            'theme': "light"
        })

    # Theme handling
    theme = st.sidebar.selectbox("Select Theme", ["light", "dark"], 
                                index=0 if st.session_state.theme == "light" else 1)
    if theme != st.session_state.theme:
        st.session_state.theme = theme

    # Apply theme
    st.markdown(f"""
        <style>
            .reportview-container {{
                background-color: {"#ffffff" if st.session_state.theme == "light" else "#111111"};
            }}
            .stButton>button {{
                margin: 2px;
            }}
            .navigation {{
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin-top: 1rem;
            }}
        </style>
    """, unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.title("Settings ⚙️")
        selected_dpi = st.slider("Image Quality (DPI)", 100, 600, DEFAULT_DPI,
                               help="Lower DPI for faster processing")

    # Main interface
    st.title("📄 PDF Check Processor")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        temp_pdf_path = os.path.join("temp", uploaded_file.name)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("🚀 Start Processing"):
            output_folder = create_unique_folder("temp_pdf_folders", uploaded_file.name)
            progress_bar = st.progress(0, text="Starting conversion...")

            try:
                start_time = time.time()
                image_paths = process_pdf_with_progress(
                    temp_pdf_path,
                    output_folder,
                    selected_dpi,
                    progress_bar
                )

                st.session_state.processed_pdfs[output_folder] = {
                    'paths': image_paths,
                    'name': uploaded_file.name
                }

                st.session_state.current_image = 0
                st.success(f"Processed {len(image_paths)} pages in {time.time() - start_time:.2f}s")

            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
            finally:
                progress_bar.empty()
                os.remove(temp_pdf_path)

    # Image viewer and navigation
    if st.session_state.processed_pdfs:
        selected_folder = st.selectbox("Select PDF", 
                                     options=list(st.session_state.processed_pdfs.keys()),
                                     format_func=lambda x: st.session_state.processed_pdfs[x]['name'])

        current_pdf = st.session_state.processed_pdfs[selected_folder]

        # Navigation controls
   

        col1, col2 = st.columns(2)
        with col1:
            if st.button("◀ Previous", disabled=st.session_state.current_image == 0):
                st.session_state.current_image -= 1
        with col2:
            if st.button("Next ▶", disabled=st.session_state.current_image == len(current_pdf['paths']) - 1):
                st.session_state.current_image += 1

        # Image display
        img_path = current_pdf['paths'][st.session_state.current_image]
        st.image(Image.open(img_path), 
                caption=f"Page {st.session_state.current_image + 1} of {len(current_pdf['paths'])}",
                use_column_width=True)

        # Progress indicator
        progress = (st.session_state.current_image + 1) / len(current_pdf['paths'])
        st.progress(progress, text=f"Navigation Progress: {progress * 100:.0f}%")

if __name__ == "__main__":
    # Create required directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("temp_pdf_folders", exist_ok=True)
    main()