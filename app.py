import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import os

# Load the brain tumor prediction model
braintumor_model = load_model('brain_tumor_binary.h5')

# Configuring Streamlit
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
st.set_page_config(
    page_title="Brain Tumor Prediction App",
    page_icon=":brain:",
    layout="wide"  # Make the layout wider
)

# Add custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            max-width: 1200px;
            margin: auto;
        }

        .title {
            color: #4285F4;
            text-align: center;
            font-size: 36px;
            margin-bottom: 20px;
        }

        .upload-btn-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }

        .btn {
            border: 2px solid gray;
            color: gray;
            background-color: white;
            padding: 8px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
        }

        .upload-btn-wrapper input[type=file] {
            font-size: 100px;
            position: absolute;
            left: 0;
            top: 0;
            opacity: 0;
        }

        .example-images {
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }

        .example-img {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            margin-right: 10px;
        }

        .prediction {
            text-align: center;
            font-size: 24px;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.markdown("<div class='title'>Brain Tumor Prediction App</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"], key="fileuploader")

    # Display example images horizontally
    example_images = [
        'examples/1 no.jpeg',
        'examples/2 no.jpeg',
        'examples/3 no.jpg',
        'examples/1 yes.jpg',
        'examples/2 yes.jpg',
        'examples/3 yes.jpg'
    ]

    st.markdown("<div class='example-images'>", unsafe_allow_html=True)
    for example in example_images:
        st.image(example, caption=f"Example: {os.path.basename(example)}", width=150, use_column_width=False, output_format='auto')
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Read the contents of the uploaded file
        file_contents = uploaded_file.read()

        # Save the uploaded file
        filename = secure_filename(uploaded_file.name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)

        with open(file_path, "wb") as f:
            f.write(file_contents)

        # Make prediction
        result = predict_braintumor(file_path)

        # Display prediction
        st.markdown("<div class='prediction'>", unsafe_allow_html=True)
        st.success(result)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
