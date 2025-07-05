import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("AgroMAT_dataset_with_references.csv")

# Preprocess
X = df.drop(columns=["ID", "Suggested Material", "Notes", "Reference Source"])
y = df["Suggested Material"]

label_encoders = {}
for col in X.columns:
    le = LabelEncoder().fit(X[col])
    X[col] = le.transform(X[col])
    label_encoders[col] = le

target_encoder = LabelEncoder().fit(y)
y_encoded = target_encoder.transform(y)

model = DecisionTreeClassifier().fit(X, y_encoded)

# UI
st.title("🌾 AgroMAT - Smart Material Recommender")
st.markdown("👩‍🌾 Helping farmers choose better materials based on field use.")

app = st.selectbox("Application", df["Application"].unique())
env = st.selectbox("Environment", df["Environment"].unique())
curr = st.selectbox("Current Material", df["Current Material"].unique())
fail = st.selectbox("Failure Mode", df["Failure Mode"].unique())
budget = st.selectbox("Budget", df["Budget"].unique())
life = st.slider("Life Expectancy (yrs)", 0.0, 10.0, 2.0)
eco = st.selectbox("Eco Priority", df["Eco Priority"].unique())

if st.button("🔍 Predict"):
    input_data = {
        "Application": app,
        "Environment": env,
        "Current Material": curr,
        "Failure Mode": fail,
        "Budget": budget,
        "Life Expectancy (yrs)": life,
        "Eco Priority": eco
    }

    encoded = [label_encoders[col].transform([input_data[col]])[0] for col in X.columns]
    pred = model.predict([encoded])[0]
    material = target_encoder.inverse_transform([pred])[0]

    st.success(f"✅ Recommended Material: {material}")
