import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import pickle

# Loading Models
braintumor_model = load_model('brain_tumor_binary.h5')

# Load the trained model
diabetes_model = pickle.load(open('diabetes.sav', 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

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

# Function to preprocess the image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(200, 200))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

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
        bloodpressure = st.number_input("Blood Pressure", 0, 122, 70)
        skinthickness = st.number_input("Skin Thickness", 0, 99, 23)
        insulin = st.number_input("Insulin", 0, 846, 79)
        bmi = st.number_input("BMI", 0.0, 67.1, 32.0)
        diabetespedigree = st.number_input("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
        age = st.number_input("Age", 21, 81, 29)

        if st.button("Predict"):
            result = predict_diabetes(pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age)
            st.success(f"Diabetes Prediction: {'Positive' if result == 1 else 'Negative'}")

if __name__ == "__main__":
    main()
