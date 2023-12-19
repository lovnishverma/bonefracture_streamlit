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
st.set_page_config(page_title="Brain Tumor Prediction App", page_icon=":brain:")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

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

def main():
    st.title("Brain Tumor Prediction App")

    # Display example images horizontally
    example_images = [
        'examples/1 no.jpeg',
        'examples/2 no.jpeg',
        'examples/3 no.jpg',
        'examples/1 yes.jpg',
        'examples/2 yes.jpg',
        'examples/3 yes.jpg'
    ]

    example_col = st.columns(len(example_images))
    for example in example_images:
        example_col[example_images.index(example)].image(example, caption=f"Example: {os.path.basename(example)}", use_column_width=True)

    clicked_example = st.selectbox("Choose an example for prediction:", example_images)

    if clicked_example is not None:
        st.image(clicked_example, caption=f"Selected Example: {os.path.basename(clicked_example)}", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Make prediction for the clicked example
        result = predict_braintumor(clicked_example)

        # Display prediction
        st.subheader("Prediction:")
        st.success(result)

if __name__ == "__main__":
    main()
