import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Finance Anomaly Detector", layout="wide")
st.title("Personal Finance Anomaly Detector")
st.write("Upload a CSV file containing your transactions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    if "Amount" not in df.columns:
        st.error("Your CSV must contain an 'Amount' colmun.")
    else:
        df["Amount"] = df["Amount"].astype(float)
        df["Abs_amount"] = df["Amount"].abs()
        df["is_debit"] = df["Amount"].apply(lambda x: 1 if x < 0 else 0)
        df["is_credit"] = df["Amount"].apply(lambda x: 1 if x > 0 else 0)
        
        features = ["Amount", "Abs_amount", "is_debit", "is_credit"]
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X_scaled)

        df["anomaly"] = model.predict(X_scaled)
        df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})
        
        st.subheader("Anomaly Detetcion Results")
        st.dataframe(df)

        anomalies = df[df["anomaly"] == "Anomaly"]

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies)

        st.subheader("Anomaly Visualization")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Index vs Amount")

            fig1, ax1 = plt.subplots(figsize=(6, 4))
            normal = df[df["anomaly"] == "Normal"]
            anomaly = df[df["anomaly"] == "Anomaly"]

            ax1.scatter(normal.index, normal["Amount"])
            ax1.scatter(anomaly.index, anomaly["Amount"])
            ax1.set_xlabel("Transaction Index")
            ax1.set_ylabel("Amount")
            ax1.set_title("Index vs Amount")

            st.pyplot(fig1)

        with col2:
            st.markdown("### Over Time")

            df["Date"] = pd.to_datetime(df["Date"])

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(df["Date"], df["Amount"])
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Amount")
            ax2.set_title("Amount Over Time")

            st.pyplot(fig2)

        st.success("Detection complete.")