import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
DATA_FOLDER = os.path.join(BASE_DIR, "data")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def image_analysis_ui():
    st.title("Image Analysis for Autism Detection")
    model_path = os.path.join(MODELS_FOLDER, "autism_model.h5")
    if not os.path.exists(model_path):
        st.error(f"Model file autism_model.h5 not found in {MODELS_FOLDER}!")
        return
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model autism_model.h5: {str(e)}")
        return
    uploaded_file = st.file_uploader("Upload an image (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])
    yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
    no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'
    if uploaded_file:
        try:
            temp_dir = os.path.join(BASE_DIR, "temp_uploads")
            os.makedirs(temp_dir, exist_ok=True)
            img_path = os.path.join(temp_dir, uploaded_file.name)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_display = Image.open(img_path)
            img_array = preprocess_image(img_path)
            prediction = model.predict(img_array)[0][0]
            if prediction > 0.015:
                st.markdown(no_style, unsafe_allow_html=True)
                result = "Non Autistic"
            else:
                st.markdown(yes_style, unsafe_allow_html=True)
                result = "Autistic"
            st.write(f"Prediction value: {prediction}")
            # Store image analysis data in pdf_text
            image_data = f"Image Analysis Results:\nFile: {uploaded_file.name}\nPrediction: {result}\nPrediction Value: {prediction}\n\n"
            st.session_state.pdf_text += image_data
            os.remove(img_path)
        except Exception as e:
            pass

if __name__ == "__main__":
    image_analysis_ui()