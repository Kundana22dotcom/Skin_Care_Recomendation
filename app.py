import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ✅ Load model and encoders (with correct full paths)
model = joblib.load("C:/Users/Dell/Downloads/pro/model_rf.pkl")
label_encoders = joblib.load("C:/Users/Dell/Downloads/pro/label_encoders.pkl")
mlb = joblib.load("C:/Users/Dell/Downloads/pro/mlb_skin_concerns.pkl")

# ✅ Set page config
st.set_page_config(page_title="SkinAI - Smart Skincare Assistant", layout="centered")

# ✅ Title and Header
st.markdown("<h1 style='text-align: center;'>🌿 SkinAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Your Smart Skincare Satisfaction Predictor</h4>", unsafe_allow_html=True)
st.write("---")

# ✅ Sidebar
st.sidebar.header("🧾 About SkinAI")
st.sidebar.markdown("""
SkinAI is a machine learning-powered assistant that predicts user satisfaction 
with skincare products based on your skin type, tone, climate, and more.
""")

st.sidebar.markdown("👤 Developed by: **KUNDANA**")
st.sidebar.markdown("📅 Project Date: July 2025")

# ✅ Input Form
with st.form("input_form"):
    st.subheader("🧍 User Profile")
    age = st.slider("Age", 15, 70, 30)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    skin_type = st.selectbox("Skin Type", label_encoders['Skin Type'].classes_)
    skin_tone = st.selectbox("Skin Tone", label_encoders['Skin Tone'].classes_)
    climate = st.selectbox("Climate", label_encoders['Climate'].classes_)

    st.subheader("🧴 Product Details")
    product_used = st.selectbox("Product Used", label_encoders['Product Used'].classes_)
    soap_brand = st.selectbox("Soap/Brand Used", label_encoders['Soap/Brand Used'].classes_)
    product_category = st.selectbox("Product Category", label_encoders['Product Category'].classes_)

    st.subheader("❗ Skin Concerns")
    all_concerns = mlb.classes_
    selected_concerns = st.multiselect("Select all that apply", all_concerns)

    submit_btn = st.form_submit_button("🔍 Predict Satisfaction")

# ✅ On Submit
if submit_btn:
    # Encode inputs
    input_data = {
        'Age': age,
        'Gender': label_encoders['Gender'].transform([gender])[0],
        'Skin Type': label_encoders['Skin Type'].transform([skin_type])[0],
        'Skin Tone': label_encoders['Skin Tone'].transform([skin_tone])[0],
        'Climate': label_encoders['Climate'].transform([climate])[0],
        'Product Used': label_encoders['Product Used'].transform([product_used])[0],
        'Soap/Brand Used': label_encoders['Soap/Brand Used'].transform([soap_brand])[0],
        'Product Category': label_encoders['Product Category'].transform([product_category])[0],
    }

    # Encode skin concerns
    concerns_encoded = mlb.transform([selected_concerns])[0]
    features = list(input_data.values()) + list(concerns_encoded)

    # Make prediction
    prediction = model.predict([features])[0]

    st.write("## 🔮 Prediction Result")
    st.success(f"🎯 Predicted Satisfaction Score: **{prediction} / 5**")

    if prediction >= 4:
        st.info("👍 This product is likely to suit your profile!")
    elif prediction == 3:
        st.warning("⚠️ This product might be average for you.")
    else:
        st.error("👎 You might not be satisfied with this product.")

# ✅ Footer
st.write("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ using Kundana</p>",
    unsafe_allow_html=True
)
