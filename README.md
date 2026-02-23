# 💰 Personal Finance Anomaly Detector

A machine learning powered web application that detects unusual financial transactions using Isolation Forest.

Built with Python, Streamlit, and Scikit-learn.

---

## 🚀 Live Demo
[Click here to try the app](https://finance-anomaly-detector-ksihdigpqzzjhbfqudngqb.streamlit.app)

---

## 📌 Project Overview

This project allows users to upload their financial transaction data (CSV format) and automatically detects anomalous transactions using an unsupervised machine learning model.

The goal is to identify unusual spending or income patterns.

---

## 🧠 How It Works

1. User uploads a CSV file containing:
   - Date
   - Description
   - Amount

2. Feature Engineering:
   - Absolute transaction amount
   - Debit indicator
   - Credit indicator

3. Feature Scaling:
   - StandardScaler is applied to normalize feature influence

4. Model:
   - Isolation Forest detects anomalies
   - Adjustable contamination parameter controls sensitivity

5. Visualization:
   - Transaction-level anomaly scatter plot
   - Daily net spending time-series plot
   - Financial summary metrics

---

## 📊 Features

- Upload custom CSV transaction data
- Adjustable anomaly sensitivity (1%–20%)
- Financial summary dashboard
- Transaction-level anomaly detection
- Daily spending trend analysis
- Deployed using Streamlit Cloud

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib

---

## 📷 Screenshot

<img width="1280" height="605" alt="Screenshot 2026-02-23 at 5 32 04 AM" src="https://github.com/user-attachments/assets/f903eee7-4698-4bb4-9add-dd2543365566" />

<img width="949" height="631" alt="Screenshot 2026-02-23 at 5 32 36 AM" src="https://github.com/user-attachments/assets/eb375558-427b-40bb-820b-2abbc4f09bff" />

<img width="936" height="599" alt="Screenshot 2026-02-23 at 5 32 51 AM" src="https://github.com/user-attachments/assets/62a27ab3-2d30-4e93-af9e-96de668a7081" />


---

## 🔍 Sample CSV Format

Date,Description,Amount
2026-02-01,Swiggy,-1200
2026-02-02,Salary,4500
2026-02-03,Rent,-8000


---

## 🎯 Key Learning Outcomes

- Implemented unsupervised anomaly detection
- Applied feature engineering techniques
- Performed feature scaling for model stability
- Built interactive ML dashboard
- Deployed ML app to production

---

## 📂 Repository Structure

finance-anomaly-detector/
│── app.py
│── requirements.txt
│── sample.csv


---

## 👨‍💻 Author

Izz Al Din Noufel Mukthar

