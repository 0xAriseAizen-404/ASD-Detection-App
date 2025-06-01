import os
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydub import AudioSegment
import io
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

base_dir = 'C:/Users/mahes/OneDrive/Desktop/FinalYearProjects/Projects/ASD-Detection-App/app/'
MODEL_PATH = base_dir + "models/lstm_asd_model.h5"
N_MFCC = 40
MAX_LEN = 100  # Number of time steps

# Load trained LSTM model
@st.cache_resource
def load_lstm_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        return None
    return load_model(MODEL_PATH)

model = load_lstm_model()

# Function to extract MFCC features
def extract_mfcc(audio_bytes):
    """Extracts MFCC features from audio file and ensures correct shape for LSTM input."""
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="m4a")
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15 - 1)  # Normalize
        sr = audio.frame_rate

        mfcc = librosa.feature.mfcc(y=samples, sr=sr, n_mfcc=N_MFCC)

        # Pad or truncate to ensure consistent shape (40 MFCCs × 100 time steps)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return np.expand_dims(mfcc.T, axis=0).astype(np.float32)  # Transpose to (1, 100, 40)

    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
        return None

# Streamlit UI
st.title("LSTM-Based Audio Analysis for ASD Detection")
st.write("Upload an **.m4a** audio file to analyze speech patterns and predict ASD likelihood.")

uploaded_file = st.file_uploader("Upload an audio file", type=["m4a"])

yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'

if uploaded_file and model:
    with st.spinner("Processing audio..."):
        mfcc_features = extract_mfcc(uploaded_file.read())  # Ensure (1, 100, 40)

        if mfcc_features is not None:
            prediction = model.predict(mfcc_features)[0]
            asd_probability = prediction[0] * 100  # Probability of ASD

            if asd_probability > 50:
                st.markdown(yes_style, unsafe_allow_html=True)
                result = "Autistic"
            else:
                st.markdown(no_style, unsafe_allow_html=True)
                result = "Non Autistic"

            st.write(f"Confidence Score: **{asd_probability:.2f}%**")

            # Store results in session for report generation
            audio_data = f"Audio Analysis Results:\nFile: {uploaded_file.name}\nModel: LSTM\nPrediction: {result}\nConfidence: {asd_probability:.2f}%\n\n"
            st.session_state.pdf_text += audio_data
        else:
            st.error("Could not extract valid MFCC features from the audio file.")
