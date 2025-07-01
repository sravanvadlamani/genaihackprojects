import streamlit as st
import pandas as pd
import datetime as dt
import os
from dotenv import load_dotenv
import openai

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Training Insights", layout="wide")
st.title("📊 Training Insights Dashboard for Managers")

# --- Load CSV from predefined file path ---
csv_file_path = "./training_data.csv"

try:
    df = pd.read_csv(csv_file_path, parse_dates=["CompletionDate", "DueDate"])
except Exception as e:
    st.error(f"❌ Error loading CSV file: {e}")
    st.stop()

# --- Upload CSV ---
#uploaded_file = st.file_uploader("Upload Training Data CSV", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file, parse_dates=["CompletionDate", "DueDate"])

# --- Preprocessing ---
df["Days Overdue"] = (pd.Timestamp.now().normalize() - pd.to_datetime(df["DueDate"])).dt.days
df["Overdue"] = (df["Status"] == "Overdue")

st.subheader("📈 Overview Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", df["EmployeeID"].nunique())
col2.metric("Completed Trainings", df[df["Status"] == "Completed"].shape[0])
col3.metric("Overdue Trainings", df[df["Status"] == "Overdue"].shape[0])

# --- Filter Section ---
with st.expander("🔍 Filter"):
    selected_team = st.multiselect("Filter by Team", df["Team"].unique())
    selected_role = st.multiselect("Filter by Role", df["Role"].unique())
    if selected_team:
        df = df[df["Team"].isin(selected_team)]
    if selected_role:
        df = df[df["Role"].isin(selected_role)]

# --- Manual Insights ---
st.subheader("📌 Actionable Observations")
overdue_by_team = df[df["Overdue"]].groupby("Team").size().sort_values(ascending=False)
low_score = df[df["Score"] < 70].dropna()

if not overdue_by_team.empty:
    st.warning("⚠️ **Teams with Most Overdue Trainings:**")
    for team, count in overdue_by_team.items():
        st.markdown(f"- **{team}**: {count} overdue trainings")

if not low_score.empty:
    st.info("📉 **Employees Scoring Below 70%:**")
    st.dataframe(low_score[["Name", "Team", "TrainingName", "Score"]])

if df[df["Status"] == "Completed"]["Score"].mean() < 75:
    st.error("⚠️ Average training score is below 75%. Consider reinforcing training materials.")

# --- Charts ---
st.subheader("📊 Training Completion Status")
st.bar_chart(df["Status"].value_counts())

st.subheader("🚨 Overdue Trainings by Role")
overdue_by_role = df[df["Overdue"]].groupby("Role").size()
if not overdue_by_role.empty:
    st.bar_chart(overdue_by_role)

# --- Download Button ---
with st.expander("⬇️ Download Processed Data"):
    st.download_button("Download as CSV", df.to_csv(index=False), "processed_training_data.csv")

# --- GPT Summary ---
def generate_summary_with_openai(df):
    overdue_count = df[df["Status"] == "Overdue"].shape[0]
    completed_count = df[df["Status"] == "Completed"].shape[0]
    low_score_df = df[df["Score"] < 70].dropna()
    avg_score = df["Score"].mean()

    prompt = f"""
    You are an HR insights assistant. Based on this summary:
    - {completed_count} trainings were completed
    - {overdue_count} are still overdue
    - {low_score_df.shape[0]} employees scored under 70%
    - The average score is {avg_score:.1f}%

    Provide 3–4 actionable insights for a manager, using clear bullet points.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error generating summary: {e}"

st.subheader("🧠 GPT-Generated Summary")
if st.button("Generate Manager Insights"):
    summary = generate_summary_with_openai(df)
    st.markdown(summary)

# else:
#     st.info("📁 Please upload a training data CSV to begin.")