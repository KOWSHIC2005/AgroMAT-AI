import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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

# ------------------------- UI STARTS HERE ---------------------------- #

st.set_page_config(page_title="AgroMAT", page_icon="🌾")
st.markdown("<h1 style='text-align: center; color: green;'>🌾 AgroMAT - புத்திசாலி மெட்டீரியல் பரிந்துரை கருவி</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>👨‍🌾 Smart, Eco-friendly & Affordable Material Selection Tool for Farmers</h4>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("### 📥 பயனர் உள்ளீடு / User Inputs")

app = st.selectbox("📦 பயன்பாடு (Application)", df["Application"].unique())
env = st.selectbox("🌤️ சூழ்நிலை (Environment)", df["Environment"].unique())
curr = st.selectbox("🔩 தற்போதைய பொருள் (Current Material)", df["Current Material"].unique())
fail = st.selectbox("💥 பாதிப்பு விதம் (Failure Mode)", df["Failure Mode"].unique())
budget = st.selectbox("💰 செலவுத்திறன் (Budget)", df["Budget"].unique())
life = st.slider("📅 ஆயுள் (வருடங்கள்) / Life Expectancy", 0.0, 10.0, 2.0)
eco = st.selectbox("♻️ பசுமை முன்னிலை (Eco Priority)", df["Eco Priority"].unique())

st.markdown("---")
if st.button("🔍 பரிந்துரை பெறு / Get Recommendation"):

    input_data = {
        "Application": app,
        "Environment": env,
        "Current Material": curr,
        "Failure Mode": fail,
        "Budget": budget,
        "Life Expectancy (yrs)": life,
        "Eco Priority": eco
    }

    encoded_input = []
    for col in X.columns:
        if col in label_encoders:
            encoded_input.append(label_encoders[col].transform([input_data[col]])[0])
        else:
            encoded_input.append(input_data[col])  # For numeric columns

    pred = model.predict([encoded_input])[0]
    material = target_encoder.inverse_transform([pred])[0]

    st.success(f"✅ பரிந்துரைக்கப்பட்ட பொருள் (Recommended Material): **{material}**")

    result_dict = input_data.copy()
    result_dict["Suggested Material"] = material
    result_df = pd.DataFrame([result_dict])

    st.download_button(
        label="⬇️ பரிந்துரை CSV ஆக பதிவிறக்க / Download as CSV",
        data=result_df.to_csv(index=False),
        file_name="AgroMAT_Prediction.csv",
        mime="text/csv"
    )

st.markdown("---")
st.markdown("🔬 <i>AgroMAT is designed by integrating Machine Learning with Materials Science, focusing on farmers' practical problems.</i>", unsafe_allow_html=True)
st.markdown("🌱 <b>Developed by Kowshic K T </b>", unsafe_allow_html=True)
