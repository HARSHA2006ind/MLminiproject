import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Diabetes Predictor")

st.title("🩺 Blood Glucose Level Classification")
st.write("Enter patient details below:")

# -------------------------------
# 🔹 NEW: Gender Selection
# -------------------------------
gender = st.selectbox("Select Gender", ["Male", "Female"])

preg = 0  # default

# 🔹 Show pregnancy ONLY for female
if gender == "Female":
    preg_option = st.selectbox("Are you pregnant?", ["No", "Yes"])
    
    if preg_option == "Yes":
        preg = st.number_input("Pregnancies", 0, 20, 1)
    else:
        preg = 0
else:
    preg = 0  # for male → always 0

# -------------------------------
# Inputs (rest same)
# -------------------------------
age = st.number_input("Age", 1, 120, 30)
glucose = st.number_input("Glucose Level", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):

    user_data = pd.DataFrame(
        [[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=X.columns
    )

    result = model.predict(user_data)

    st.subheader("🧾 Result:")

    # Range-based output
    if glucose < 100:
        st.success("✅ Safe (Normal Range)")
    elif 100 <= glucose < 140:
        st.warning("⚠️ Prediabetic (Need Attention)")
    else:
        st.error("🚨 Diabetic (High Risk)")

    # -------------------------------
    # 🔹 Personalized Message
    # -------------------------------
    st.subheader("💡 Personalized Insight")

    if gender == "Male":
        st.info("General health monitoring recommended for male patients.")

    elif gender == "Female":
        if preg_option == "Yes":
            st.info("Pregnancy detected → Regular glucose monitoring is very important.")
        else:
            st.info("General health monitoring recommended for female patients.")

# -------------------------------
# 📊 Horizontal Graph
# -------------------------------
st.subheader("📊 Glucose Level Categories")

labels = ["Safe (<100)", "Prediabetic (100-140)", "Diabetic (>140)"]
values = [100, 140, 200]

fig, ax = plt.subplots()

ax.barh(labels, values)
ax.set_xlabel("Glucose Level")
ax.set_title("Glucose Range Classification")

st.pyplot(fig)