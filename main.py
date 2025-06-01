import streamlit as st
from chatbot import chatbot_ui
from quiz import quiz_ui
from games import games_ui
from audio_analysis import audio_analysis_ui
from image_analysis import image_analysis_ui

# ⛔️ This must be before ANY other st.* call
st.set_page_config(page_title="ASD Detection App", layout="wide")

# Initialize session state for pdf_text if not already present
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.title("🔴 ASD Detection App")
    with st.expander("⭐ Navigation", expanded=True):
        st.session_state.page = st.radio(
            "Go to",
            ["Home", "Quiz Test", "Gaming Tests", "Audio Analysis", "Image Analysis", "Chatbot"],
            index=["Home", "Quiz Test", "Gaming Tests", "Audio Analysis", "Image Analysis", "Chatbot"].index(st.session_state.page)
        )

# Function to render the next page button
def render_next_button():
    page_order = ["Home", "Quiz Test", "Gaming Tests", "Audio Analysis", "Image Analysis", "Chatbot"]
    current_idx = page_order.index(st.session_state.page)
    if current_idx < len(page_order) - 1:
        next_page = page_order[current_idx + 1]
        button_label = f"Go to {next_page}"
        if next_page == "Quiz Test":
            button_label = "Start Quiz Test"
        elif next_page == "Gaming Tests":
            button_label = "Start Behavioural Games Test"
        elif next_page == "Audio Analysis":
            button_label = "Start Audio Analysis"
        elif next_page == "Image Analysis":
            button_label = "Start Image Analysis"
        elif next_page == "Chatbot":
            button_label = "Go to Chatbot and Analyse Data"
        
        if st.button(button_label):
            st.session_state.page = next_page
            st.rerun()

# Home Page (Landing Page)
if st.session_state.page == "Home":
    st.title("Welcome to the ASD Detection App")
    st.markdown("""
    ### Autism Spectrum Disorder (ASD) Notice
    Autism Spectrum Disorder (ASD) is a developmental condition that affects communication, behavior, and social interaction in varying degrees. Early detection and intervention can significantly improve outcomes. Our app is designed to assist in screening and understanding ASD through innovative tools and games.

    #### Features of Our App
    - **Quiz Test**: Answer a series of questions to assess potential ASD traits using a trained XGBoost model.
    - **Gaming Tests**: Engage in interactive games (Emoji Recognition, Memory, Ball Clicker) to evaluate cognitive and behavioral patterns.
    - **Audio Analysis**: Upload audio recordings to analyze vocal patterns for ASD indicators using machine learning models.
    - **Image Analysis**: Submit images to detect autism-related features with a pre-trained neural network.
    - **Chatbot**: Get personalized analysis and support by uploading reports or asking questions.

    #### Precautions and Suggestions
    - **Consult Professionals**: This app is a screening tool, not a diagnostic replacement. Always consult a healthcare professional for a formal diagnosis.
    - **Privacy**: Ensure personal data is handled securely and not shared without consent.
    - **Engagement**: Encourage participation in all tests for a comprehensive screening experience.
    - **Awareness**: Learn about ASD symptoms like delayed speech, repetitive behaviors, or social challenges to better understand the condition.

    Start your journey with us below!
    """)
    render_next_button()

elif st.session_state.page == "Quiz Test":
    quiz_ui()
    render_next_button()

elif st.session_state.page == "Gaming Tests":
    games_ui()
    render_next_button()

elif st.session_state.page == "Audio Analysis":
    audio_analysis_ui()
    render_next_button()

elif st.session_state.page == "Image Analysis":
    image_analysis_ui()
    render_next_button()

elif st.session_state.page == "Chatbot":
    chatbot_ui()