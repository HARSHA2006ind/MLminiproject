import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Predictions for evaluation
y_pred = model.predict(X_test)

# -------------------------------
# Metrics
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confidence (sample)
probs = model.predict_proba(X_test)
confidence = np.max(probs[0])

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Diabetes Predictor")

st.title("🩺 Blood Glucose Level Classification")
st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age", 1, 120, 30)
glucose = st.number_input("Glucose Level", 0, 200, 120)
bp = st.number_input("Blood Pressure", 0, 150, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
preg = st.number_input("Pregnancies", 0, 20, 1)

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
        st.warning("⚠️ Prediabetic")
    else:
        st.error("🚨 Diabetic")

    # Model prediction
    if result[0] == 1:
        st.error("Model Prediction: Diabetic")
    else:
        st.success("Model Prediction: Not Diabetic")

# -------------------------------
# Metrics Display
# -------------------------------
st.subheader("📊 Model Performance")

st.write(f"Accuracy: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")
st.write(f"Confidence (sample): {confidence:.2f}")

# -------------------------------
# Confusion Matrix
# -------------------------------
st.subheader("📉 Confusion Matrix")

fig_cm, ax_cm = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
disp.plot(ax=ax_cm)

st.pyplot(fig_cm)

# -------------------------------
# Bar Chart
# -------------------------------
st.subheader("📈 Performance Chart")

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]

fig, ax = plt.subplots()
ax.bar(metrics, values)
ax.set_ylabel("Score")
ax.set_title("Model Performance Metrics")

st.pyplot(fig)

# -------------------------------
# Horizontal Glucose Graph
# -------------------------------
st.subheader("📊 Glucose Level Categories")

labels = ["Safe (<100)", "Prediabetic (100-140)", "Diabetic (>140)"]
values = [100, 140, 200]

fig2, ax2 = plt.subplots()
ax2.barh(labels, values)
ax2.set_xlabel("Glucose Level")
ax2.set_title("Glucose Range Classification")

st.pyplot(fig2)