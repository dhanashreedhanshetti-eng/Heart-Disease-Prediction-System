import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load("heart_disease_model.pkl")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

html, body, [class*="css"]{
    font-family: 'Segoe UI';
}

.block-container{
    padding-top:2rem;
}

.stButton>button{
    width:100%;
    height:60px;
    background:#dc2626;
    color:white;
    font-size:22px;
    border-radius:12px;
    border:none;
    font-weight:bold;
}

.stButton>button:hover{
    background:#b91c1c;
    color:white;
}

div[data-testid="stMetric"]{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.image(
    "https://img.icons8.com/color/96/heart-with-pulse.png",
    width=90
)

st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("---")

st.sidebar.success("""
### 👩‍💻 Developer

**Dhanashree Dhanshetti**
""")

st.sidebar.info("""
### 🤖 Machine Learning Model

**Algorithm:** Random Forest Classifier

**Accuracy:** 83.60%
""")

st.sidebar.markdown("---")

st.sidebar.write("CodeAlpha Machine Learning Internship")

# ---------------- HEADER ----------------

st.markdown(
"""
<h1 style='text-align:center;color:#ef4444;'>
❤️ Heart Disease Prediction System
</h1>
""",
unsafe_allow_html=True
)

st.markdown("""
<div style="
background:#1e293b;
padding:25px;
border-radius:18px;
border-left:8px solid #ef4444;
color:white;
box-shadow:0px 5px 20px rgba(0,0,0,0.3);
">

<h2>🏥 AI Medical Decision Support System</h2>

<p style="font-size:18px;">

Predicts the probability of Heart Disease using a
<b>Random Forest Machine Learning Model</b>
trained on the
<b>UCI Heart Disease Dataset.</b>

</p>

</div>
""", unsafe_allow_html=True)

st.write("")

left, right = st.columns(2)

# ---------------- LEFT COLUMN ----------------

with left:

    st.subheader("👤 Patient Information")

    age = st.number_input(
        "Age",
        min_value=20,
        max_value=100,
        value=50,
        help="Patient age in years."
    )

    sex = st.selectbox(
        "Gender",
        ["Female","Male"],
        help="Select patient's gender."
    )

    sex_value = 1 if sex=="Male" else 0

    cp_options = {
        "Typical Angina":0,
        "Atypical Angina":1,
        "Non-anginal Pain":2,
        "Asymptomatic":3
    }

    cp = st.selectbox(
        "Chest Pain Type",
        list(cp_options.keys()),
        help="Type of chest pain experienced."
    )

    cp_value = cp_options[cp]

    trestbps = st.number_input(
        "Resting Blood Pressure (mmHg)",
        80,
        250,
        120,
        help="Normal resting BP is around 120 mmHg."
    )

    chol = st.number_input(
        "Cholesterol (mg/dL)",
        100,
        600,
        200,
        help="Normal cholesterol is below 200 mg/dL."
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar >120 mg/dL",
        ["No","Yes"],
        help="Is fasting blood sugar greater than 120 mg/dL?"
    )

    fbs_value = 1 if fbs=="Yes" else 0

    # ---------------- RIGHT COLUMN ----------------

with right:

    st.subheader("🩺 Clinical Information")

    restecg_options = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }

    restecg_name = st.selectbox(
        "Rest ECG",
        list(restecg_options.keys()),
        help="Electrocardiogram result while resting."
    )

    restecg = restecg_options[restecg_name]

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        60,
        220,
        150,
        help="Highest heart rate achieved during exercise."
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"],
        help="Chest pain experienced during exercise."
    )

    exang_value = 1 if exang == "Yes" else 0

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        6.5,
        1.0,
        help="ST depression induced by exercise relative to rest."
    )

    slope_options = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }

    slope_name = st.selectbox(
        "Slope of ST Segment",
        list(slope_options.keys()),
        help="Slope of the peak exercise ST segment."
    )

    slope = slope_options[slope_name]

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4],
        help="Number of major blood vessels detected by fluoroscopy."
    )

    thal_options = {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2,
        "Unknown": 3
    }

    thal_name = st.selectbox(
        "Thallium Stress Test",
        list(thal_options.keys()),
        help="Result of the Thallium Stress Test."
    )

    thal = thal_options[thal_name]

st.write("")

predict = st.button("❤️ Predict Heart Disease")

# ---------------- PREDICTION ----------------

if predict:

    data = np.array([[
        age,
        sex_value,
        cp_value,
        trestbps,
        chol,
        fbs_value,
        restecg,
        thalach,
        exang_value,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(data)
    confidence = model.predict_proba(data)

    st.markdown("---")

    if prediction[0] == 1:

        score = confidence[0][1]

        st.error("## ⚠️ Prediction Result : HIGH RISK OF HEART DISEASE")

        st.progress(float(score))

        st.metric(
            "Prediction Confidence",
            f"{score*100:.2f}%"
        )

        st.info("""
### 🩺 Recommendation

✔ Consult a Cardiologist

✔ Monitor Blood Pressure Regularly

✔ Reduce Cholesterol Levels

✔ Exercise Under Medical Guidance

✔ Eat a Heart-Healthy Diet

✔ Avoid Smoking & Alcohol
""")

    else:

        score = confidence[0][0]

        st.success("## ✅ Prediction Result : LOW RISK OF HEART DISEASE")

        st.progress(float(score))

        st.metric(
            "Prediction Confidence",
            f"{score*100:.2f}%"
        )

        st.balloons()

        st.success("""
### 🌿 Healthy Lifestyle Recommendation

✔ Continue Regular Exercise

✔ Eat Nutritious Food

✔ Maintain Healthy Weight

✔ Monitor Blood Pressure

✔ Have Regular Health Check-ups

Keep taking care of your heart ❤️
""")

# ---------------- DISCLAIMER ----------------

st.markdown("---")

st.warning("""
### ⚠️ Medical Disclaimer

This prediction is generated using a Machine Learning model trained on the UCI Heart Disease Dataset.

It is intended **only for educational and demonstration purposes**.

It is **NOT** a substitute for professional medical diagnosis, treatment, or advice.

Always consult a qualified healthcare professional for any medical concerns.
""")

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown("""
<center>

<h4>❤️ Developed by <b>Dhanashree Dhanshetti</b></h4>

Machine Learning Intern • CodeAlpha

Random Forest Classifier | Python | Streamlit

© 2026 All Rights Reserved

</center>
""", unsafe_allow_html=True)