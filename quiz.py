import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Load questions from JSON in data folder
def load_questions():
    project_root = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(project_root, "data", "asd-questions.json")
    try:
        with open(questions_path, "r") as f:
            questions_data = json.load(f)
        return questions_data["questions"]
    except FileNotFoundError:
        st.error(f"Questions file (asd-questions.json) not found at {questions_path}.")
        return []

# Load dataset from CSV in data folder
def load_dataset():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_root, "data", "dataset-asd.csv")
    try:
        df = pd.read_csv(dataset_path)
        return df
    except FileNotFoundError:
        st.error(f"Dataset file (dataset-asd.csv) not found at {dataset_path}.")
        return None

def train_xgboost_model(df):
    if df is None or df.empty:
        st.error("No dataset available to train the model.")
        return None
    features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 'Qchat-10-Score', 
                'Sex', 'Jaundice', 'Family_mem_with_ASD', 'Ethnicity']
    df['Sex'] = df['Sex'].map({'m': 1, 'f': 0})
    df['Jaundice'] = df['Jaundice'].map({'yes': 1, 'no': 0})
    df['Family_mem_with_ASD'] = df['Family_mem_with_ASD'].map({'yes': 1, 'no': 0})
    ethnicity_map = {
        'asian': 0, 'black': 1, 'hispanic': 2, 'latino': 3, 'middle eastern': 4, 'mixed': 5, 
        'native indian': 6, 'pacifica': 7, 'south asian': 8, 'white european': 9, 'unknown': 10
    }
    df['Ethnicity'] = df['Ethnicity'].str.lower().map(ethnicity_map).fillna(10)
    X = df[features]
    y = df['Class/ASD Traits '].map({'Yes': 1, 'No': 0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model

def predict_autism(user_answers, xgb_model, df):
    if xgb_model is None or df is None:
        st.error("Model or dataset not available for prediction.")
        return None, None
    answers = {f'A{i+1}': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(ans) 
               for i, ans in enumerate(user_answers[:9])}
    answers['A10'] = ["Clear and early", "Delayed but clear", "Delayed and unclear", "Still not speaking"].index(user_answers[9])
    answers['Age_Mons'] = int(user_answers[10]) * 12 if user_answers[10].isdigit() else 0
    answers['Qchat-10-Score'] = sum(answers[f'A{i+1}'] for i in range(9)) + answers['A10']
    answers['Sex'] = 1 if user_answers[11].lower() == "male" else 0
    answers['Jaundice'] = 1 if user_answers[12].lower() == "yes" else 0
    answers['Family_mem_with_ASD'] = 1 if user_answers[13].lower() == "yes" else 0
    ethnicity_map = {
        'asian': 0, 'black': 1, 'hispanic': 2, 'latino': 3, 'middle eastern': 4, 'mixed': 5, 
        'native indian': 6, 'pacifica': 7, 'south asian': 8, 'white european': 9, 'unknown': 10
    }
    answers['Ethnicity'] = ethnicity_map.get(user_answers[14].lower(), 10)
    input_data = pd.DataFrame([answers])
    prediction = xgb_model.predict(input_data)[0]
    probability = xgb_model.predict_proba(input_data)[0][1]
    return prediction, probability

def save_to_dataset(user_answers, prediction):
    df = load_dataset()
    if df is None:
        return
    new_case_no = df['Case_No'].max() + 1 if not df.empty else 1
    new_row = {
        'Case_No': new_case_no,
        'A1': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[0]),
        'A2': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[1]),
        'A3': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[2]),
        'A4': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[3]),
        'A5': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[4]),
        'A6': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[5]),
        'A7': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[6]),
        'A8': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[7]),
        'A9': ["Always", "Usually", "Sometimes", "Rarely", "Never"].index(user_answers[8]),
        'A10': ["Clear and early", "Delayed but clear", "Delayed and unclear", "Still not speaking"].index(user_answers[9]),
        'Age_Mons': int(user_answers[10]) * 12 if user_answers[10].isdigit() else 0,
        'Qchat-10-Score': sum(["Always", "Usually", "Sometimes", "Rarely", "Never"].index(ans) for ans in user_answers[:9]) + ["Clear and early", "Delayed but clear", "Delayed and unclear", "Still not speaking"].index(user_answers[9]),
        'Sex': 'm' if user_answers[11].lower() == "male" else 'f',
        'Ethnicity': user_answers[14].lower(),
        'Jaundice': 'yes' if user_answers[12].lower() == "yes" else 'no',
        'Family_mem_with_ASD': 'yes' if user_answers[13].lower() == "yes" else 'no',
        'Who completed the test': user_answers[15].lower(),
        'Class/ASD Traits ': 'Yes' if prediction == 1 else 'No'
    }
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset-asd.csv"), index=False)

def generate_pdf_report(user_answers, prediction, probability, questions):
    additional_questions = [
        "What is the age of your child (in years)?",
        "What is the gender of your child?",
        "Was your child born with jaundice?",
        "Does any immediate family member have a history with ASD?",
        "What is the ethnicity of your child?",
        "Who is completing the test?"
    ]
    all_questions = [q["question"] for q in questions] + additional_questions
    context = f"""
    Autism Screening Report:
    User Answers: {list(zip(all_questions, user_answers))}
    Prediction: {'Autism' if prediction == 1 else 'No Autism'}
    Probability of Autism: {probability:.2f}
    """
    response = model.generate_content(context)
    report_text = response.text if response else "Unable to generate report content."
    if "Autism Screening Report" in report_text:
        report_content = report_text.split("Autism Screening Report:")[1].strip()
    else:
        report_content = report_text
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    styles['Title'].fontSize = 20
    styles['Title'].fontName = 'Helvetica-Bold'
    styles['Heading1'].fontSize = 16
    styles['Heading1'].fontName = 'Helvetica-Bold'
    styles['Normal'].fontSize = 12
    styles['Normal'].fontName = 'Helvetica'
    styles['Normal'].leading = 14
    elements = [
        Paragraph("Autism Screening Report", styles['Title']),
        Spacer(1, 12),
        Paragraph(f"<font size=18><b>{'Autism' if prediction == 1 else 'No Autism'}</b></font>", styles['Heading1']),
        Spacer(1, 12),
        Paragraph(f"Probability of Autism: <b>{probability:.2f}</b>", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Report Details", styles['Heading1']),
        Spacer(1, 6)
    ]
    for line in report_content.split("\n"):
        if line.strip():
            if "screening tools are not diagnostic" in line.lower():
                line = f"<b>Screening Tools Are Not Diagnostic:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "false positives are possible" in line.lower():
                line = f"<b>False Positives Are Possible:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "context is missing" in line.lower():
                line = f"<b>Context Is Missing:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "age plays a role" in line.lower():
                line = f"<b>Age Plays a Role:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elif "next steps" in line.lower():
                line = f"<b>Next Steps:</b> {line.split(':', 1)[1] if ':' in line else line}"
            elements.append(Paragraph(line, styles['Normal']))
            elements.append(Spacer(1, 6))
    elements.append(Paragraph("User Answers:", styles['Heading1']))
    elements.append(Spacer(1, 6))
    for i, (question, answer) in enumerate(zip(all_questions, user_answers), 1):
        answer_text = f"Q{i}: {question} - <b>{answer}</b>"
        elements.append(Paragraph(answer_text, styles['Normal']))
        elements.append(Spacer(1, 4))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def quiz_ui():
    st.title("Autism Quiz Test")
    questions = load_questions()
    df = load_dataset()
    if not questions or df is None:
        return
    xgb_model = train_xgboost_model(df)
    user_answers = []
    st.subheader("Please answer the following questions:")
    for q in questions:
        answer = st.selectbox(q["question"], q["options"], key=f"q{q['id']}")
        user_answers.append(answer)
    age = st.text_input("What is the age of your child (in years)?", key="age")
    gender = st.selectbox("What is the gender of your child?", ["Male", "Female"], key="gender")
    jaundice = st.selectbox("Was your child born with jaundice?", ["Yes", "No"], key="jaundice")
    family_history = st.selectbox("Does any immediate family member have a history with ASD?", ["Yes", "No"], key="family_history")
    ethnicity = st.selectbox("What is the ethnicity of your child?", 
                             ["Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Mixed", 
                              "Native Indian", "Pacifica", "South Asian", "White European", "Unknown"], 
                             key="ethnicity")
    completed_by = st.selectbox("Who is completing the test?", 
                                ["Family Member", "Self", "Mother", "Health Care Professional", "Others"], 
                                key="completed_by")
    user_answers.extend([age, gender, jaundice, family_history, ethnicity, completed_by])
    if st.button("Submit Quiz"):
        if all(len(ans) > 0 for ans in user_answers[:9]) and age:
            prediction, probability = predict_autism(user_answers, xgb_model, df)
            if prediction is not None:
                st.session_state.prediction = prediction
                st.session_state.probability = probability
                if prediction == 0:
                    st.success(f"Result: No Autism (Probability of Autism: {probability:.2f})", icon="✅")
                else:
                    st.error(f"Result: Autism (Probability of Autism: {probability:.2f})", icon="⚠️")
                save_to_dataset(user_answers, prediction)
                # Store quiz data in pdf_text
                additional_questions = [
                    "What is the age of your child (in years)?",
                    "What is the gender of your child?",
                    "Was your child born with jaundice?",
                    "Does any immediate family member have a history with ASD?",
                    "What is the ethnicity of your child?",
                    "Who is completing the test?"
                ]
                all_questions = [q["question"] for q in questions] + additional_questions
                quiz_data = f"Quiz Test Results:\nUser Answers: {list(zip(all_questions, user_answers))}\nPrediction: {'Autism' if prediction == 1 else 'No Autism'}\nProbability: {probability:.2f}\n\n"
                st.session_state.pdf_text += quiz_data
        else:
            st.error("Please answer all questions before submitting.")
    if st.button("Generate Report"):
        if 'prediction' in st.session_state and 'probability' in st.session_state:
            pdf_buffer = generate_pdf_report(user_answers, st.session_state.prediction, st.session_state.probability, questions)
            st.download_button(
                label="Download PDF Report",
                data=pdf_buffer,
                file_name="autism_screening_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Please submit the quiz first to generate a report.")

if __name__ == "__main__":
    quiz_ui()