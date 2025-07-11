import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("AgroMAT_dataset_with_references.csv")

# Preprocess inputs
X = df.drop(columns=["ID", "Suggested Material", "Notes", "Reference Source"])
y = df["Suggested Material"]

# Label encode categorical columns
label_encoders = {}
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder().fit(X[col])
        X[col] = le.transform(X[col])
        label_encoders[col] = le

# Encode the target
target_encoder = LabelEncoder().fit(y)
y_encoded = target_encoder.transform(y)

# Train the model
model = DecisionTreeClassifier().fit(X, y_encoded)

# ------------------------- UI STARTS ---------------------------- #

st.set_page_config(page_title="AgroMAT", page_icon="🌾")
st.title("🌾 AgroMAT - Smart Material Recommender")
st.markdown("👨‍🌾 Helping farmers choose durable, eco-friendly, and affordable materials for agriculture-related applications.")
st.markdown("---")

st.subheader("📥 User Inputs")

app = st.selectbox("Application", df["Application"].unique())
env = st.selectbox("Environment", df["Environment"].unique())
curr = st.selectbox("Current Material", df["Current Material"].unique())
fail = st.selectbox("Failure Mode", df["Failure Mode"].unique())
budget = st.selectbox("Budget", df["Budget"].unique())
life = st.slider("Life Expectancy (years)", 0.0, 10.0, 2.0)
eco = st.selectbox("Eco Priority", df["Eco Priority"].unique())

st.markdown("---")
if st.button("🔍 Predict Best Material"):

    input_data = {
        "Application": app,
        "Environment": env,
        "Current Material": curr,
        "Failure Mode": fail,
        "Budget": budget,
        "Life Expectancy (yrs)": life,
        "Eco Priority": eco
    }

    # Encode input
    encoded_input = []
    for col in X.columns:
        if col in label_encoders:
            encoded_input.append(label_encoders[col].transform([input_data[col]])[0])
        else:
            encoded_input.append(input_data[col])

    pred = model.predict([encoded_input])[0]
    material = target_encoder.inverse_transform([pred])[0]

    st.success(f"✅ Recommended Material: **{material}**")

    result_dict = input_data.copy()
    result_dict["Suggested Material"] = material
    result_df = pd.DataFrame([result_dict])

    st.download_button(
        label="⬇️ Download Prediction as CSV",
        data=result_df.to_csv(index=False),
        file_name="AgroMAT_Prediction.csv",
        mime="text/csv"
    )

st.caption(" Built by Kowshic K T ")
