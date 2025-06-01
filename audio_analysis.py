import os
import numpy as np
import librosa
import joblib
import streamlit as st
from pydub import AudioSegment
import io
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ann.pkl")

def extract_mfcc(audio_data, sample_rate, n_mfcc=20):
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    return mfcc_features

def audio_analysis_ui():
    st.title("Audio Analysis for Autism Detection (ANN Model)")
    
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        return
    
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    uploaded_file = st.file_uploader("Upload an audio file (.m4a)", type="m4a")

    yes_style = '<h1 style="color:red;text-align:center;">Prediction: Autistic</h1>'
    no_style = '<h1 style="color:green;text-align:center;">Prediction: Non Autistic</h1>'

    if uploaded_file:
        try:
            audio = AudioSegment.from_file(io.BytesIO(uploaded_file.getvalue()), format='m4a')
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15 - 1)
            sr = audio.frame_rate
            mfcc_features = extract_mfcc(samples, sr)

            if not np.isnan(mfcc_features).any():
                mfcc_avg = np.mean(mfcc_features, axis=1, keepdims=True)
                mfcc_features_reshaped = mfcc_avg.reshape(1, -1)

                predicted_label = model.predict(mfcc_features_reshaped)
                result = "Autistic" if predicted_label[0] == 1 else "Non Autistic"
                style = yes_style if result == "Autistic" else no_style
                st.markdown(style, unsafe_allow_html=True)

                # Ensure session variable exists
                if "pdf_text" not in st.session_state:
                    st.session_state.pdf_text = ""

                # Save result in session for report
                audio_data = (
                    f"Audio Analysis Results:\n"
                    f"File: {uploaded_file.name}\n"
                    f"Prediction: {result}\n\n"
                )
                st.session_state.pdf_text += audio_data
                # st.write(st.session_state.pdf_text)

            else:
                st.error("Could not extract valid MFCC features from the audio file.")

        except Exception as e:
            st.error(f"Failed to process the audio file: {e}")

if __name__ == "__main__":
    audio_analysis_ui()
