import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to preprocess a set of images
def preprocess_images(img_set):
    processed_set = []
    for img_path in img_set:
        processed_set.append(preprocess_image(img_path))
    return np.vstack(processed_set)

# Function to predict brain tumor probability
def predict_tumor_probability(model, img_set):
    processed_set = preprocess_images(img_set)
    return model.predict(processed_set)

# Sidebar
st.sidebar.title("Brain Tumor Detection App")

# Upload images
st.sidebar.header("Upload Images")
uploaded_files = st.sidebar.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Example images
example_images = [
    'examples/1 no.jpeg',
    'examples/2 no.jpeg',
    'examples/3 no.jpg',
    'examples/1 yes.jpg',
    'examples/2 yes.jpg',
    'examples/3 yes.jpg'
]

# Display examples
st.sidebar.header("Example Images")
selected_examples = st.sidebar.multiselect("Select example images:", example_images)

# Combine uploaded and selected examples
selected_images = uploaded_files or selected_examples

if selected_images:
    st.sidebar.header("Selected Images")
    st.sidebar.image(selected_images, width=100)

    # Predict tumor probability
    st.header("Brain Tumor Detection Results")
    st.subheader("Uploaded and Selected Images")

    # Load the model
    braintumor_model = load_model('models/brain_tumor_binary.h5')

    # Display prediction results
    probabilities = predict_tumor_probability(braintumor_model, selected_images)

    for i, img_path in enumerate(selected_images):
        st.image(img_path, caption=f"Probability: {probabilities[i][0]:.2%}", use_column_width=True)

else:
    st.warning("Please upload or select at least one image.")

