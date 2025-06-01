import streamlit as st
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_games_pdf_report(game_data):
    emoji_data = game_data.get("emoji_data", {})
    memory_data = game_data.get("memory_data", {})
    ball_clicker_data = game_data.get("ball_clicker_data", {})
    emoji_summary = "Not played"
    memory_summary = "Not played"
    ball_clicker_summary = "Not played"
    emoji_trails_summary = "No trails"
    memory_trails_summary = "No trails"
    ball_clicker_trails_summary = "No trails"
    if emoji_data:
        latest_trail = max(emoji_data.items(), key=lambda x: x[0], default=(None, {"score": "Not played", "total": 0, "accuracy": 0, "status": 0})) if emoji_data else (None, {"score": "Not played", "total": 0, "accuracy": 0, "status": 0})
        if latest_trail[1]["score"] != "Not played":
            emoji_summary = f"Score: {latest_trail[1]['score']}/{latest_trail[1]['total']}, Accuracy: {latest_trail[1]['accuracy']:.2f}%"
        emoji_trails_summary = ", ".join([f"Trail {k}: Score {v['score']}/{v['total']}, Accuracy {v['accuracy']:.2f}%" for k, v in emoji_data.items()]) or "No trails"
    if memory_data:
        latest_trail = max(memory_data.items(), key=lambda x: x[0], default=(None, {"score": "Not played", "miss": 0, "total": 0, "accuracy": 0, "status": 0})) if memory_data else (None, {"score": "Not played", "miss": 0, "total": 0, "accuracy": 0, "status": 0})
        if latest_trail[1]["score"] != "Not played":
            memory_summary = f"Score: {latest_trail[1]['score']}, Misses: {latest_trail[1]['miss']}, Accuracy: {latest_trail[1]['accuracy']:.2f}%"
        memory_trails_summary = ", ".join([f"Trail {k}: Score {v['score']}, Misses {v['miss']}, Accuracy {v['accuracy']:.2f}%" for k, v in memory_data.items()]) or "No trails"
    if ball_clicker_data:
        latest_trail = max(ball_clicker_data.items(), key=lambda x: x[0], default=(None, {"score": "Not played", "miss": 0, "total": 0, "accuracy": 0, "status": 0})) if ball_clicker_data else (None, {"score": "Not played", "miss": 0, "total": 0, "accuracy": 0, "status": 0})
        if latest_trail[1]["score"] != "Not played":
            ball_clicker_summary = f"Score: {latest_trail[1]['score']}, Misses: {latest_trail[1]['miss']}, Accuracy: {latest_trail[1]['accuracy']:.2f}%"
        ball_clicker_trails_summary = ", ".join([f"Trail {k}: Score {v['score']}, Misses {v['miss']}, Accuracy {v['accuracy']:.2f}%" for k, v in ball_clicker_data.items()]) or "No trails"
    context = f"""
    Autism Screening Games Report:
    Emoji Recognition Test:
    - Latest Result: {emoji_summary}
    - All Trails: {emoji_trails_summary}
    Memory Test:
    - Latest Result: {memory_summary}
    - All Trails: {memory_trails_summary}
    Ball Clicker Test:
    - Latest Result: {ball_clicker_summary}
    - All Trails: {ball_clicker_trails_summary}
    """
    response = model.generate_content(context)
    report_text = response.text if response else "Unable to generate report content."
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Autism Screening Games Report", styles['Title']),
        Spacer(1, 12),
        Paragraph("Game Results:", styles['Heading1']),
        Spacer(1, 6),
        Paragraph(f"Emoji Recognition Test: {emoji_summary}", styles['Normal']),
        Spacer(1, 4),
        Paragraph(f"Memory Test: {memory_summary}", styles['Normal']),
        Spacer(1, 4),
        Paragraph(f"Ball Clicker Test: {ball_clicker_summary}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("All Trails:", styles['Heading1']),
        Spacer(1, 6),
        Paragraph(f"Emoji Recognition Test: {emoji_trails_summary}", styles['Normal']),
        Spacer(1, 4),
        Paragraph(f"Memory Test: {memory_trails_summary}", styles['Normal']),
        Spacer(1, 4),
        Paragraph(f"Ball Clicker Test: {ball_clicker_trails_summary}", styles['Normal']),
        Spacer(1, 12),
        Paragraph("Analysis:", styles['Heading1']),
        Spacer(1, 6),
    ]
    for line in report_text.split("\n"):
        if line.strip():
            elements.append(Paragraph(line, styles['Normal']))
            elements.append(Spacer(1, 6))
    doc.build(elements)
    buffer.seek(0)
    return buffer

def load_game_html(game_file):
    project_root = os.path.dirname(os.path.abspath(__file__))
    game_path = os.path.join(project_root, "assets", game_file)
    try:
        with open(game_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Game file {game_file} not found.")
        return ""

def fetch_local_storage_data(game_type):
    fetch_js = f"""
    <script>
        function fetchLocalStorageData() {{
            const data = JSON.parse(localStorage.getItem("{game_type}_game_data")) || {{}};
            Streamlit.setComponentValue({{ game: "{game_type}", data: data }});
        }}
        fetchLocalStorageData();
    </script>
    """
    return fetch_js

def games_ui():
    st.title("Gaming Tests")
    if "game_data" not in st.session_state:
        st.session_state.game_data = {
            "emoji_data": {},
            "memory_data": {},
            "ball_clicker_data": {}
        }
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main"
    if st.session_state.current_page == "main":
        st.write("Select a test to begin:")
        if st.button("Emoji Recognition Test"):
            st.session_state.current_page = "emoji_game"
        if st.button("Memory Test"):
            st.session_state.current_page = "memory_game"
        if st.button("Ball Clicker Test"):
            st.session_state.current_page = "ball_clicker_game"
    elif st.session_state.current_page == "emoji_game":
        st.subheader("Emoji Recognition Test")
        if st.button("Get back to Games"):
            st.session_state.current_page = "main"
        html_content = load_game_html("emoji_game.html")
        if html_content:
            st.components.v1.html(html_content, height=1000, scrolling=True)
    elif st.session_state.current_page == "memory_game":
        st.subheader("Memory Test")
        if st.button("Get back to Games"):
            st.session_state.current_page = "main"
        html_content = load_game_html("memory_game.html")
        if html_content:
            st.components.v1.html(html_content, height=1000, scrolling=True)
    elif st.session_state.current_page == "ball_clicker_game":
        st.subheader("Ball Clicker Test")
        if st.button("Get back to Games"):
            st.session_state.current_page = "main"
        html_content = load_game_html("ball_clicker_game.html")
        if html_content:
            st.components.v1.html(html_content, height=1000, scrolling=True)

if __name__ == "__main__":
    games_ui()