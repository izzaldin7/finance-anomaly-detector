import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

        st.sidebar.header("Model Settings")

        contamination = st.sidebar.slider("Anomaly Sensitivity (%)", min_value = 1, max_value = 50, value = 5) / 100

        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(X_scaled)

        df["anomaly"] = model.predict(X_scaled)
        df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

        st.subheader("Financial Summary")

        total_debit = df[df["Amount"] < 0]["Amount"].sum()
        total_credit = df[df["Amount"] > 0]["Amount"].sum()
        anomaly_count = df[df["anomaly"] == "Anomaly"].shape[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Debit", f"{total_debit:.2f}")
        col2.metric("Total Credit", f"{total_credit:.2f}")
        col3.metric("Anomalies Detected", anomaly_count)
        
        st.subheader("Anomaly Detetcion Results")
        st.dataframe(df)

        anomalies = df[df["anomaly"] == "Anomaly"]

        st.subheader("Detected Anomalies")
        st.dataframe(anomalies)

        st.subheader("Visual Analysis")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        fig, col = plt.subplots(1, 2, figsize=(16, 6))

        col[0].scatter(df.index, df["Amount"], alpha=0.6)

        anomaly_points = df[df["anomaly"] == "Anomaly"]
        col[0].scatter(anomaly_points.index, anomaly_points["Amount"], s=100)

        col[0].set_title("Anomaly Detection (Index vs Amount)")
        col[0].set_xlabel("Transaction Index")
        col[0].set_ylabel("Amount")

        daily = df.groupby("Date")["Amount"].sum().reset_index()

        col[1].plot(daily["Date"], daily["Amount"])

        col[1].set_title("Daily Net Spending")
        col[1].set_xlabel("Date")
        col[1].set_ylabel("Net Amount")

        col[1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
        col[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.setp(col[1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        st.pyplot(fig)

        st.success("Detection complete.")