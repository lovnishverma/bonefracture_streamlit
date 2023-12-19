import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import pickle

# Loading Models
braintumor_model = load_model('models/brain_tumor_binary.h5')

# Load the trained model
diabetes_model = pickle.load(open('models/diabetes.sav', 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_imgs(set_name, img_size):
    # Implementation

def crop_imgs(set_name, add_pixels_value=0):
    # Implementation

# Function to preprocess the image
def preprocess_image(file_path):
    # Implementation

def predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    pred = diabetes_model.predict([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]])
    return pred

def predict_braintumor(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_processed = crop_imgs([img_gray])
    img_processed = preprocess_imgs(img_processed, (224, 224))
    pred = braintumor_model.predict(img_processed)

    # Handle binary decision
    return 1 if pred[0] >= 0.5 else 0

def main():
    st.title("Medical Diagnosis App")

    menu = ["Home", "Brain Tumor", "Diabetes"]
    choice = st.sidebar.selectbox("Select Option", menu)

    if choice == "Home":
        st.subheader("Home Page")
        st.text("Welcome to the Medical Diagnosis App!")

    elif choice == "Brain Tumor":
        st.subheader("Brain Tumor Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Save the uploaded file
            file_path = "temp_image.jpg"
            uploaded_file.save(file_path)

            # Make prediction
            result = predict_braintumor(file_path)

            st.success(f"Prediction: {'Positive' if result == 1 else 'Negative'}")

    elif choice == "Diabetes":
        st.subheader("Diabetes Prediction")
        pregnancies = st.number_input("Pregnancies", 0, 17, 1)
        glucose = st.number_input("Glucose", 0, 200, 100)
        # Add other input fields as needed

        if st.button("Predict"):
            result = predict_diabetes(pregnancies, glucose, ..., ...)  # Add other input variables
            st.success(f"Diabetes Prediction: {'Positive' if result == 1 else 'Negative'}")

if __name__ == "__main__":
    main()
