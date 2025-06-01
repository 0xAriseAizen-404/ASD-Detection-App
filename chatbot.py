import streamlit as st
import fitz
import os
import io
import warnings
from dotenv import load_dotenv
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def query_deepseek(messages):
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=messages,
            # extra_headers={
            #     "HTTP-Referer": "https://your-site-url.com",
            #     "X-Title": "ASD Detection App"
            # },
            extra_body={}
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"[ERROR] DeepSeek API: {e}"

def handle_chat_input(prompt):
    if prompt:
        st.session_state.chat_history.append(("User", prompt))
        messages = [
            {"role": "system", "content": "You are an expert assistant specialized in Autism Spectrum Disorder (ASD). Answer user questions professionally and empathetically."}
        ] + [
            {"role": role.lower(), "content": msg} for role, msg in st.session_state.chat_history
        ]

        if st.session_state.pdf_text:
            messages.append({"role": "system", "content": f"Collected Data:\n{st.session_state.pdf_text}"})

        messages.append({"role": "user", "content": prompt})
        bot_response = query_deepseek(messages)
        st.session_state.chat_history.append(("Bot", bot_response))

def process_pdf_upload():
    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state.pdf_text = pdf_text
        st.session_state.chat_history.append(("Bot", "PDF successfully uploaded and analyzed."))

        context = f"""
You are a medical analyst specialized in Autism Spectrum Disorder (ASD). 

You have the following patient's test prediction result:

{st.session_state.pdf_text}

Please provide a clear, compassionate clinical interpretation focused only on the patient.

Focus on:
- Precautions and recommendations for patient care.
- Next steps for diagnosis and support.

Write in terms of the patient and simply and clearly for parents or caregivers.
Use bullet points and headings.
"""
        messages = [
            {"role": "system", "content": "You are a helpful ASD medical assistant."},
            {"role": "user", "content": context}
        ]
        bot_response = query_deepseek(messages)
        st.session_state.chat_history.append(("Bot", bot_response))

def generate_comprehensive_pdf_report():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Comprehensive ASD Screening Report", styles['Title']),
        Spacer(1, 12),
        Paragraph("Collected Data Analysis", styles['Heading1']),
        Spacer(1, 6)
    ]
    if st.session_state.pdf_text:
        context = f"Collected Data for Analysis:\n{st.session_state.pdf_text}\n\nProvide a detailed analysis and recommendations."
        messages = [
            {"role": "system", "content": "You are an expert ASD medical assistant."},
            {"role": "user", "content": context}
        ]
        analysis = query_deepseek(messages)
        elements.append(Paragraph("Analysis:", styles['Heading1']))
        elements.append(Spacer(1, 6))
        for line in analysis.split("\n"):
            if line.strip():
                elements.append(Paragraph(line, styles['Normal']))
                elements.append(Spacer(1, 4))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Raw Data:", styles['Heading1']))
        elements.append(Spacer(1, 6))
        for line in st.session_state.pdf_text.split("\n"):
            if line.strip():
                elements.append(Paragraph(line, styles['Normal']))
                elements.append(Spacer(1, 4))
    else:
        elements.append(Paragraph("No data collected yet from previous pages.", styles['Normal']))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def chatbot_ui():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    st.title("Autism Support Chatbot")
    chat_container = st.container()
    with chat_container:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role.lower()):
                st.write(message)

    st.subheader("Upload Your ASD Report")
    st.file_uploader("", type="pdf", key="pdf_uploader", on_change=process_pdf_upload)

    prompt = st.chat_input("Type your message here...")
    if prompt:
        handle_chat_input(prompt)
        st.rerun()

    if st.button("Generate Comprehensive Report"):
        pdf_buffer = generate_comprehensive_pdf_report()
        st.download_button(
            label="Download Comprehensive Report",
            data=pdf_buffer,
            file_name="comprehensive_asd_report.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    chatbot_ui()
