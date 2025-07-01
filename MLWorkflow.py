import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import openai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="Training AI Insights", layout="centered")
st.title("🧠 Training Risk Insights")

# Load training data from fixed path
csv_file_path = "./training_data.csv"
try:
    df = pd.read_csv(csv_file_path, parse_dates=["CompletionDate", "DueDate"])
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# Feature engineering
df["DaysUntilDue"] = (df["DueDate"] - pd.Timestamp.now().normalize()).dt.days
df["Overdue"] = df["Status"] == "Overdue"
df["Completed"] = df["Status"] == "Completed"

# Encode features
features = df[["Team", "Role", "TrainingName", "DaysUntilDue"]].copy()
target = df["Overdue"].astype(int)

le_team = LabelEncoder()
le_role = LabelEncoder()
le_training = LabelEncoder()
features["Team"] = le_team.fit_transform(features["Team"])
features["Role"] = le_role.fit_transform(features["Role"])
features["TrainingName"] = le_training.fit_transform(features["TrainingName"])

# Train simple model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
df["Overdue_Prob"] = model.predict_proba(features)[:, 1]
df["Predicted_Overdue"] = model.predict(features)

# --- GPT Summary ---
@st.cache_data(show_spinner=False)
def generate_summary(df):
    completed = df[df["Completed"]].shape[0]
    overdue = df[df["Overdue"]].shape[0]
    predicted_high_risk = df[df["Overdue_Prob"] > 0.7].shape[0]
    avg_score = df["Score"].mean()
    low_score_df = df[df["Score"] < 70].dropna()

    prompt = f"""
    You are an HR training insights assistant.

    Based on this data:
    - {completed} trainings were completed
    - {overdue} are currently overdue
    - {predicted_high_risk} trainings are predicted to be overdue with >70% confidence
    - {low_score_df.shape[0]} employees scored under 70%
    - The average score is {avg_score:.1f}%

    Provide 3–4 bullet point insights for a manager to act on.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error generating GPT summary: {e}"

# Display summary by default
st.subheader("📋 AI-Generated Summary (Auto)")
with st.spinner("Generating insights..."):
    summary = generate_summary(df)
st.markdown(summary)

# Ask if user wants details
if st.button("🔍 Show Detailed Predictions"):
    st.subheader("📊 ML-Based Overdue Risk Prediction")
    st.write("Trainings most at risk of becoming overdue:")
    st.dataframe(
        df[[
            "Name", "Team", "Role", "TrainingName", "DueDate", "Status", "Overdue_Prob", "Predicted_Overdue"
        ]].sort_values("Overdue_Prob", ascending=False).reset_index(drop=True)
    )
else:
    st.info("Click 'Show Detailed Predictions' to explore risk scores and predicted overdue trainings.")
