import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import openai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import tempfile

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Training Insights", layout="wide")
st.title("🧠 Training Risk Insights Dashboard")

# --- Load and preprocess training data ---
csv_path = "./training_data.csv"
try:
    df = pd.read_csv(csv_path, parse_dates=["CompletionDate", "DueDate"])
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

df["DaysUntilDue"] = (df["DueDate"] - pd.Timestamp.now().normalize()).dt.days
df["Overdue"] = df["Status"] == "Overdue"
df["Completed"] = df["Status"] == "Completed"

# --- Feature encoding ---
features = df[["Team", "Role", "TrainingName", "DaysUntilDue"]].copy()
target = df["Overdue"].astype(int)

le_team = LabelEncoder()
le_role = LabelEncoder()
le_training = LabelEncoder()
features["Team"] = le_team.fit_transform(features["Team"])
features["Role"] = le_role.fit_transform(features["Role"])
features["TrainingName"] = le_training.fit_transform(features["TrainingName"])

# --- Train ML model ---
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Predict overdue risk ---
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
    Based on this HR training data:
    - {completed} trainings were completed
    - {overdue} are overdue
    - {predicted_high_risk} are predicted to be overdue (>70% risk)
    - {low_score_df.shape[0]} employees scored below 70%
    - Average score is {avg_score:.1f}%

    Provide 3–4 bullet-point insights a manager can act on.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response['choices'][0]['message']['content']

# --- Display GPT summary ---
st.subheader("📋 AI-Generated Summary")
st.markdown(generate_summary(df))

# --- Chat interface ---
st.subheader("💬 Ask About Your Training Data")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask a question:")
if user_query:
    context = df.head(200).to_string(index=False)[:3000]
    chat_prompt = f"""
    CSV Preview:
    {context}

    Question: {user_query}
    Answer:
    """
    try:
        reply = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": chat_prompt}],
            temperature=0.4
        )
        answer = reply.choices[0].message.content
        st.session_state.chat_history.append((user_query, answer))
    except Exception as e:
        st.error(f"❌ Error: {e}")

for q, a in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")

# --- Risk Visuals and Predictions ---
if st.button("📊 Show Risk Predictions and Visuals"):
    st.subheader("🔍 ML Predictions for Overdue Risk")
    st.dataframe(
        df[["Name", "Team", "Role", "TrainingName", "DueDate", "Status", "Overdue_Prob", "Predicted_Overdue"]]
        .sort_values("Overdue_Prob", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("📈 Training Status Breakdown")
    st.bar_chart(df["Status"].value_counts())

    st.subheader("📉 Overdue Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["Overdue_Prob"], bins=10, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of Overdue Probabilities")
    ax.set_xlabel("Predicted Overdue Probability")
    ax.set_ylabel("Number of Trainings")
    st.pyplot(fig)

    st.subheader("🚨 Top High-Risk Roles (>70% Probability)")
    high_risk_roles = df[df["Overdue_Prob"] > 0.7]["Role"].value_counts()
    if not high_risk_roles.empty:
        st.bar_chart(high_risk_roles)
    else:
        st.success("No roles with overdue risk >70%.")

# --- Export to PDF ---
def export_to_pdf(summary_text, df):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # --- Summary Section ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Training Insights Summary", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.ln(5)

    # --- Risk Table Section ---
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Top Predicted High-Risk Trainings (>70%)", ln=True)
    pdf.set_font("Arial", size=11)

    risky = df[df["Overdue_Prob"] > 0.7][["Name", "Team", "TrainingName", "DueDate", "Overdue_Prob"]]
    if risky.empty:
        pdf.cell(0, 10, "No high-risk trainings detected.", ln=True)
    else:
        for _, row in risky.iterrows():
            pdf.cell(0, 10, f"{row['Name']} | {row['Team']} | {row['TrainingName']} | Due: {row['DueDate'].date()} | Risk: {row['Overdue_Prob']:.2f}", ln=True)

    # --- Visual Charts Section ---
    def add_chart_to_pdf(fig, title):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            fig.savefig(tmp_img.name, bbox_inches="tight")
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, title, ln=True)
            pdf.image(tmp_img.name, x=10, w=190)

    # 1. Training Status
    status_fig, ax1 = plt.subplots()
    df["Status"].value_counts().plot(kind="bar", ax=ax1, color="skyblue", edgecolor="black")
    ax1.set_title("Training Status Breakdown")
    ax1.set_ylabel("Count")
    add_chart_to_pdf(status_fig, "Training Status Breakdown")

    # 2. Overdue Probability Histogram
    prob_fig, ax2 = plt.subplots()
    ax2.hist(df["Overdue_Prob"], bins=10, color="salmon", edgecolor="black")
    ax2.set_title("Overdue Probability Distribution")
    ax2.set_xlabel("Overdue Risk")
    ax2.set_ylabel("Number of Trainings")
    add_chart_to_pdf(prob_fig, "Overdue Probability Distribution")

    # 3. High-Risk Roles
    role_fig, ax3 = plt.subplots()
    high_risk_roles = df[df["Overdue_Prob"] > 0.7]["Role"].value_counts()
    if not high_risk_roles.empty:
        high_risk_roles.plot(kind="bar", ax=ax3, color="orange", edgecolor="black")
        ax3.set_title("Top High-Risk Roles")
        ax3.set_ylabel("Count")
        add_chart_to_pdf(role_fig, "Top High-Risk Roles")

    # Save PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        return tmp_pdf.name

st.subheader("📤 Export")
if st.button("Export Summary to PDF"):
    file_path = export_to_pdf(generate_summary(df), df)
    with open(file_path, "rb") as f:
        st.download_button("📄 Download PDF", data=f, file_name="training_summary.pdf", mime="application/pdf")
