import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import openai

# Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set Streamlit page
st.set_page_config(page_title="Training Insights", layout="centered")
st.title("🧠 Automated Training Insights")

# Load CSV data from predefined path
csv_file_path = "./training_data.csv"
try:
    df = pd.read_csv(csv_file_path, parse_dates=["CompletionDate", "DueDate"])
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Preprocess data
df["Days Overdue"] = (pd.Timestamp.now().normalize() - pd.to_datetime(df["DueDate"])).dt.days
df["Overdue"] = df["Status"] == "Overdue"

# Generate insights with OpenAI
@st.cache_data(show_spinner=False)
def generate_summary(df):
    completed = df[df["Status"] == "Completed"].shape[0]
    overdue = df[df["Overdue"]].shape[0]
    low_score_df = df[df["Score"] < 70].dropna()
    avg_score = df["Score"].mean()

    prompt = f"""
    You are an HR training insights assistant.

    Based on this data summary:
    - {completed} trainings were completed
    - {overdue} trainings are overdue
    - {low_score_df.shape[0]} employees scored under 70%
    - The average score is {avg_score:.1f}%

    Please provide 3 to 4 clear, actionable insights a manager can use to address team training gaps.
    Format them as bullet points.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error generating insights: {e}"

# Display generated insights
st.subheader("📋 Actionable Training Insights")
with st.spinner("Analyzing data and generating insights..."):
    insights = generate_summary(df)
st.markdown(insights)
