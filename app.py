import streamlit as st
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("Bone_fracture_classifier_model.h5")

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

# Function to preprocess the image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

def main():
    st.title("Bone Fracture Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Check if the file extension is allowed
        if allowed_file(uploaded_file.name):
            # Display the selected image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Save the uploaded image temporarily
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as temp_image:
                temp_image.write(uploaded_file.read())

            # Preprocess the image
            img_array = preprocess_image(temp_image_path)

            # Make prediction
            prediction = model.predict(img_array)[0, 0]
            result = "Broken" if prediction > 0.5 else "Not Broken"

            st.write(f"Prediction: {result}")
            st.write(f"Confidence: {prediction:.2%}")

            # Remove the temporary image file
            os.remove(temp_image_path)
        else:
            st.warning("Invalid file format. Please upload an image with a valid format (jpg, jpeg, or png).")

if __name__ == "__main__":
    main()
