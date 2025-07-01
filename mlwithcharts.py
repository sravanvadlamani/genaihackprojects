import streamlit as st
import pandas as pd
import numpy as np
import os
import openai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load OpenAI key
# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Training Insights", layout="wide")
st.title("🧠 AI Training Risk Insights")

# Load CSV from path
csv_path = "./training_data.csv"
try:
    df = pd.read_csv(csv_path, parse_dates=["CompletionDate", "DueDate"])
except Exception as e:
    st.error(f"❌ Could not load training data: {e}")
    st.stop()

# --- Preprocessing ---
df["DaysUntilDue"] = (df["DueDate"] - pd.Timestamp.now().normalize()).dt.days
df["Overdue"] = df["Status"] == "Overdue"
df["Completed"] = df["Status"] == "Completed"

# Encode categorical variables
features = df[["Team", "Role", "TrainingName", "DaysUntilDue"]].copy()
target = df["Overdue"].astype(int)

le_team = LabelEncoder()
le_role = LabelEncoder()
le_training = LabelEncoder()
features["Team"] = le_team.fit_transform(features["Team"])
features["Role"] = le_role.fit_transform(features["Role"])
features["TrainingName"] = le_training.fit_transform(features["TrainingName"])

# Train model
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
    Based on this HR training data:
    - {completed} trainings were completed
    - {overdue} are currently overdue
    - {predicted_high_risk} trainings are predicted to become overdue (prob > 70%)
    - {low_score_df.shape[0]} employees scored under 70%
    - Average score is {avg_score:.1f}%

    Generate 3–4 bullet point insights for a manager to act on.
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

# --- Show Summary by Default ---
st.subheader("📋 AI Summary (Auto)")
with st.spinner("Generating insights..."):
    st.markdown(generate_summary(df))

# --- Filters ---
with st.expander("🔍 Filter by Team and Role"):
    selected_team = st.multiselect("Select Team(s)", df["Team"].unique())
    selected_role = st.multiselect("Select Role(s)", df["Role"].unique())
    filtered_df = df.copy()
    if selected_team:
        filtered_df = filtered_df[filtered_df["Team"].isin(selected_team)]
    if selected_role:
        filtered_df = filtered_df[filtered_df["Role"].isin(selected_role)]

# --- Show Predictions Button ---
if st.button("🔍 Show Risk Predictions and Visuals"):
    st.subheader("📊 Predicted Overdue Risk")
    st.dataframe(
        filtered_df[[
            "Name", "Team", "Role", "TrainingName", "DueDate", "Status", "Overdue_Prob", "Predicted_Overdue"
        ]].sort_values("Overdue_Prob", ascending=False).reset_index(drop=True)
    )

    # --- Chart: Training Status Breakdown ---
    st.subheader("📈 Training Status Breakdown")
    status_counts = filtered_df["Status"].value_counts()
    st.bar_chart(status_counts)

    # --- Chart: Overdue Probability Distribution ---
    st.subheader("📉 Overdue Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(filtered_df["Overdue_Prob"], bins=10, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of Overdue Risk")
    ax.set_xlabel("Predicted Overdue Probability")
    ax.set_ylabel("Number of Trainings")
    st.pyplot(fig)

    # --- Chart: High Risk by Role ---
    st.subheader("🚨 High-Risk Roles (>70% Prob)")
    high_risk_roles = filtered_df[filtered_df["Overdue_Prob"] > 0.7]["Role"].value_counts()
    if not high_risk_roles.empty:
        st.bar_chart(high_risk_roles)
    else:
        st.success("No roles with high predicted risk (>70%).")
else:
    st.info("Click the button above to explore ML predictions and visual analytics.")
