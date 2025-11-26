import streamlit as st
import numpy as np
import pickle

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Liver Disease Stage Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------
st.markdown("""
 <style>
    .main-title {
        background: linear-gradient(to right, #4b79a1, #283e51);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 25px;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 600;
        text-align: center;
        margin-top: 20px;
    }
    .stApp {
        background-color: #f5f7fa !important;
    }
 </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# LOAD MODEL + SCALER + LABEL ENCODER
# ------------------------------------------------------
try:
    model = pickle.load(open('best_liver_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------
st.sidebar.title("🩺 Liver Disease Predictor")
st.sidebar.write("---")
st.sidebar.info(
    "Predicts 5 stages of liver disease:\n"
    "- No Disease\n"
    "- Suspect Disease\n"
    "- Hepatitis\n"
    "- Fibrosis\n"
    "- Cirrhosis"
)
st.sidebar.write("---")
st.sidebar.subheader("📊 Normal Range (Min–Max) Based on Dataset")

# NORMAL RANGE BOXES – FROM YOUR CSV FILE
st.sidebar.markdown("""
<div class='range-box'>
<b>Age:</b> 19 – 77<br>
<b>Albumin:</b> 14.9 – 82.2<br>
<b>Alkaline Phosphatase:</b> 11.3 – 416.6<br>
<b>ALT (Alanine Aminotransferase):</b> 0.9 – 325.3<br>
<b>AST (Aspartate Aminotransferase):</b> 10.6 – 324.0<br>
<b>Bilirubin:</b> 0.8 – 254.0<br>
<b>Cholinesterase:</b> 1.42 – 16.41<br>
<b>Cholesterol:</b> 1.43 – 9.67<br>
<b>Creatinine:</b> 8.0 – 1079.1<br>
<b>Gamma GT:</b> 4.5 – 650.9<br>
</div>
""", unsafe_allow_html=True)


st.sidebar.write("---")
st.sidebar.write("Made with ❤️ **Project Group 4**")


# ------------------------------------------------------
# MAIN TITLE
# ------------------------------------------------------
st.markdown("<div class='main-title'>Liver Disease Stage Prediction</div>", unsafe_allow_html=True)

st.write("### Provide the patient's test details below:")

# ------------------------------------------------------
# INPUT FORM
# ------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    albumin = st.number_input("Albumin", min_value=0.0, value=4.0)
    alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0.0, value=80.0)
    alanine_aminotransferase = st.number_input("Alanine Aminotransferase (ALT)", min_value=0.0, value=25.0)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", min_value=0.0, value=22.0)

with col2:
    bilirubin = st.number_input("Bilirubin", min_value=0.0, value=1.0)
    cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=6.0)
    cholesterol = st.number_input("Cholesterol", min_value=0.0, value=180.0)
    creatinina = st.number_input("Creatinine", min_value=0.0, value=1.0)
    gamma_gt = st.number_input("Gamma GT", min_value=0.0, value=20.0)
    protein = st.number_input("Protein", min_value=0.0, value=7.0)

# ------------------------------------------------------
# PREPARE FEATURES (Correct order)
# ------------------------------------------------------
features = np.array([[age, sex, albumin, alkaline_phosphatase,
                      alanine_aminotransferase, aspartate_aminotransferase,
                      bilirubin, cholinesterase, cholesterol, creatinina,
                      gamma_gt, protein]])

# ------------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------------
predict_btn = st.button("🔍 Predict Stage", use_container_width=True)

# ------------------------------------------------------
# PROCESS PREDICTION
# ------------------------------------------------------
if predict_btn:

    try:
        # Scale input
        features_scaled = scaler.transform(features)

        # Predict
        pred = model.predict(features_scaled)[0]

        # Convert using label encoder
        try:
            stage = le.inverse_transform([pred])[0]
        except:
            stage = pred  # In case already string

        # Color mapping
        color_map = {
            "no disease": "#27ae60",
            "suspect disease": "#f1c40f",
            "hepatitis": "#e67e22",
            "fibrosis": "#d35400",
            "cirrhosis": "#c0392b"
        }
        box_color = color_map.get(stage.lower(), "#3498db")

        # Display output
        st.markdown(
            f"""
            <div class='prediction-box' style='background-color: {box_color}; color: white;'>
                Predicted Stage: <b>{stage.upper()}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.stop()


# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("<br><center>© 2025 Liver Disease Predictor</center>", unsafe_allow_html=True)
