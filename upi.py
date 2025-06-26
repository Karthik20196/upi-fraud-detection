import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("fraud_model.pkl")
columns = joblib.load("model_columns.pkl")
le_dict = joblib.load("label_encoders.pkl")  # You must have saved this during training

st.set_page_config(page_title="UPI Fraud Detector", layout="centered")
st.title("💳 UPI Fraud Detection App")
st.markdown("Choose how you want to test a transaction:")

mode = st.radio("Input Mode", ["Upload CSV", "Manual Entry"])

# ================================
# 📁 CSV Upload Mode
# ================================
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.subheader("📋 Uploaded Data")
        st.write(data.head())

        if set(columns).issubset(data.columns):
            # Ensure amount is float
            data['amount'] = data['amount'].astype(float)

            # Predict
            probas = model.predict_proba(data[columns])[:, 1]
            data['Fraud Risk %'] = (probas * 100).round(2)
            data['Fraud Prediction'] = (data['Fraud Risk %'] > 50).astype(int)

            st.subheader("🔍 Prediction Results")
            st.write(data[['amount', 'app_used', 'merchant_name', 'Fraud Risk %', 'Fraud Prediction']].head(10))

            # Download button
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "fraud_predictions.csv", "text/csv")

        else:
            st.error("Uploaded file does not contain required columns")

# ================================
# 📝 Manual Entry Mode
# ================================
else:
    st.subheader("📝 Enter Transaction Details Manually")

    amount = st.number_input("Transaction Amount (₹)", min_value=1.0, max_value=20000.0)
    app_used = st.selectbox("App Used", le_dict['app_used'].classes_)
    merchant_name = st.selectbox("Merchant Name", le_dict['merchant_name'].classes_)
    receiver_verified = st.selectbox("Is Receiver Verified?", le_dict['receiver_verified'].classes_)
    payment_mode = st.selectbox("Payment Mode", le_dict['payment_mode'].classes_)

    if st.button("Check Fraud Risk"):
        input_data = pd.DataFrame([[amount, app_used, merchant_name, receiver_verified, payment_mode]],
                                  columns=['amount', 'app_used', 'merchant_name', 'receiver_verified', 'payment_mode'])

        # Encode categorical columns
        for col in ['app_used', 'merchant_name', 'receiver_verified', 'payment_mode']:
            input_data[col] = le_dict[col].transform(input_data[col])

        # Ensure amount is float
        input_data['amount'] = input_data['amount'].astype(float)

        # Predict
        prob = model.predict_proba(input_data)[:, 1][0]
        fraud_percent = round(prob * 100, 2)

        st.metric("Fraud Risk %", f"{fraud_percent}%")

        if fraud_percent > 50:
            st.error("⚠️ Fraudulent Transaction Likely!")
        else:
            st.success("✅ Transaction Seems Legitimate")
