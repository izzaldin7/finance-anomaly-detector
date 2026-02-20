import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

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

        X = df[["Amount"]]

        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)

        df["anomaly"] = model.predict(X)
        df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

        st.subheader("Anomaly Detetcion Results")
        st.dataframe(df)

        anomalies = df[df["anomaly"] == "Anomaly"]

        st.subheader("Detected Anomalies!")
        st.dataframe(anomalies)

        st.success("Detection complete.")