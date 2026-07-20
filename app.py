import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Heart Stroke Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------ CUSTOM CSS ------------------
st.markdown(
    """
    <style>
    .main {
        padding: 0 1rem;
    }
    .stApp {
        background-color: black;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        padding: 1.5rem 1.5rem 0.5rem 1.5rem;
        margin-bottom: 1rem;
    }
    .st-bw {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        padding: 1rem 1.5rem;
    }
    .result-box {
        border-radius: 12px;
        padding: 1rem 1.5rem;
        font-weight: 600;
        text-align: center;
    }
    .result-high {
        background-color: #ffe5e5;
        border-left: 6px solid #dc3545;
        color: #b02a37;
    }
    .result-low {
        background-color: #e6f5e6;
        border-left: 6px solid #28a745;
        color: #1a7a3a;
    }
    h1 {
        color: #0b2b4a;
        font-weight: 700;
    }
    .stButton button {
        background-color: #0b2b4a;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        width: 100%;
        border: none;
        transition: 0.2s;
    }
    .stButton button:hover {
        background-color: #1d4b6e;
        color: white;
        box-shadow: 0 4px 12px rgba(11,43,74,0.3);
    }
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_models()

# ------------------ HEADER ------------------
st.markdown("<h1 style='text-align: center;'>❤️ Heart Stroke Risk Assessment</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #6c757d; font-size: 1.1rem;'>"
    "Enter your health parameters below to evaluate your risk of heart disease.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ------------------ INPUT FORM ------------------
with st.form(key="prediction_form"):

    # ---- LEFT COLUMN: Personal & Blood Pressure ----
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 👤 Personal Information")
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=40,
            step=1,
            help="Your current age in years.",
        )
        sex = st.selectbox(
            "Sex",
            options=["M", "F"],
            help="Biological sex at birth.",
        )
        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=["ATA", "NAP", "TA", "ASY"],
            help=(
                "ATA: Atypical Angina, NAP: Non-Anginal Pain, "
                "TA: Typical Angina, ASY: Asymptomatic."
            ),
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=120,
            step=1,
            help="Resting blood pressure in mm Hg.",
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=200,
            step=1,
            help="Serum cholesterol in mg/dL.",
        )

    with col2:
        st.markdown("#### 🫀 Clinical & ECG Measures")
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="1 = fasting blood sugar > 120 mg/dL, 0 = normal.",
        )
        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=["Normal", "ST", "LVH"],
            help="ST = ST-T wave abnormality, LVH = Left Ventricular Hypertrophy.",
        )
        max_hr = st.slider(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
            help="Maximum heart rate during exercise (bpm).",
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            options=["Y", "N"],
            help="Y = experienced angina during exercise, N = no.",
        )
        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            min_value=0.0,
            max_value=6.0,
            value=1.0,
            step=0.1,
            help="ST depression induced by exercise relative to rest.",
        )
        st_slope = st.selectbox(
            "ST Slope",
            options=["Up", "Flat", "Down"],
            help="Slope of the peak exercise ST segment.",
        )

    # ---- SUBMIT BUTTON ----
    submitted = st.form_submit_button("🔍 Predict Your Risk")

# ------------------ PREDICTION & RESULTS ------------------
if submitted:
    # Build raw input dictionary
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Align with training columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    # ---- DISPLAY RESULT ----
    st.divider()
    st.markdown("### 📊 Prediction Result")

    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
    with col_res2:
        if prediction == 1:
            st.markdown(
                '<div class="result-box result-high">⚠️ <b>High Risk</b> – Please consult a cardiologist.</div>',
                unsafe_allow_html=True,
            )
            # progress bar (risk level)
            st.progress(0.85, text="Risk Level: High")
        else:
            st.markdown(
                '<div class="result-box result-low">✅ <b>Low Risk</b> – Keep up the healthy lifestyle!</div>',
                unsafe_allow_html=True,
            )
            st.progress(0.15, text="Risk Level: Low")

    # Optional: show input summary
    with st.expander("📋 View your input summary", expanded=False):
        st.dataframe(input_df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    st.caption("ℹ️ This prediction is for informational purposes only and does not replace professional medical advice.")

# ------------------ FOOTER ------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: #adb5bd; font-size: 0.9rem;'>"
    "Built with ❤️ using Streamlit & KNN model</p>",
    unsafe_allow_html=True,
)