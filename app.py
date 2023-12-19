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
    layout="wide"
)

# Define colors for better styling
primary_color = "#1f7ed0"
secondary_color = "#2a2a2a"
text_color = "#ffffff"
background_color = "#f4f4f4"

# Set the style for the page
st.markdown(
    f"""
    <style>
        body {{
            color: {text_color};
            background-color: {background_color};
        }}
        .reportview-container .main {{
            color: {text_color};
            background-color: {background_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Function to allow only specific file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess and crop images
def preprocess_imgs(set_name, img_size):
    set_new = []
    for img in set_name:
        img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)

def crop_imgs(set_name, add_pixels_value=0):
    set_new = []
    for img in set_name:
        gray = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
                      extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
        set_new.append(new_img)
    return np.array(set_new)

# Function to preprocess an image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make a prediction
def predict_braintumor(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Crop and preprocess the grayscale image
    img_processed = crop_imgs([img_gray])
    img_processed = preprocess_imgs(img_processed, (224, 224))

    # Make prediction
    pred = braintumor_model.predict(img_processed)

    # Handle binary decision
    confidence = pred[0][0]

    if confidence >= 0.5:
        return "Brain Tumor Not Found!"
    else:
        return "Brain Tumor Found!"

# Main function for the app
def main():
    st.title("Brain Tumor Prediction App")

    # Sidebar for uploading a new image
    st.sidebar.header("Upload New Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    # Display example images horizontally
    example_images = [
        'examples/1 no.jpeg',
        'examples/2 no.jpeg',
        'examples/3 no.jpg',
        'examples/1 yes.jpg',
        'examples/2 yes.jpg',
        'examples/3 yes.jpg'
    ]

    # Display example images in a row
    st.write("## Example Images")
    example_col = st.columns(len(example_images))
    for example in example_images:
        example_col[example_images.index(example)].image(example, caption=f"Example: {os.path.basename(example)}", use_column_width=True)

    st.write("")

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("")

        # Classify the uploaded image
        st.write("## Classifying...")
        result = predict_braintumor(uploaded_file.name)

        # Display prediction
        st.subheader("Prediction:")
        st.success(result)

    st.write("")

    # Display selected example image and prediction
    clicked_example = st.selectbox("Choose an example for prediction:", example_images)
    if clicked_example is not None:
        st.write("## Selected Example Image")
        st.image(clicked_example, caption=f"Example: {os.path.basename(clicked_example)}", use_column_width=True)
        st.write("")

        # Classify the selected example image
        st.write("## Classifying...")
        result = predict_braintumor(clicked_example)

        # Display prediction
        st.subheader("Prediction:")
        st.success(result)

if __name__ == "__main__":
    main()
