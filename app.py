# ============================ Imports ============================
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

from sklearn.model_selection import train_test_split

# ============================ Page Config ============================
st.set_page_config(
    page_title="HeartGuard AI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================ Session State ============================
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ============================ Paths ============================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"

# ============================ Safe Loaders ============================
@st.cache_resource(show_spinner=False)
def load_models_and_scaler():
    model_files = {
        "Logistic Regression": "logistic_regression_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl"
    }

    models = {}
    for name, file in model_files.items():
        path = MODEL_DIR / file
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {file}")
        models[name] = joblib.load(path)

    scaler_path = MODEL_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.pkl not found")

    scaler = joblib.load(scaler_path)
    return models, scaler


@st.cache_data(show_spinner=False)
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("heart.csv not found in data folder")
    return pd.read_csv(DATA_PATH)

# ============================ Startup ============================
try:
    models, scaler = load_models_and_scaler()
    df = load_dataset()
except Exception as e:
    st.error(f"Startup failed: {e}")
    st.stop()

# ============================ Feature Schema ============================
TARGET_COL = "target"
FEATURE_NAMES = df.drop(columns=[TARGET_COL]).columns.tolist()

# ============================ Sidebar ============================
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)
model = models[model_choice]

st.sidebar.markdown("---")
st.sidebar.caption("HeartGuard AI -- ML-based risk analysis")

# ============================ Helper ============================
def safe_index(options, value):
    return options.index(value) if value in options else 0

# ============================ User Input ============================
def user_input_form(df):
    st.subheader("Patient Information")
    inputs = {}

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        inputs["age"] = st.number_input("Age", 1, 120, int(df["age"].median()))

    with c2:
        options = [0, 1]
        inputs["sex"] = st.selectbox(
            "Sex",
            options,
            index=safe_index(options, int(df["sex"].median())),
            format_func=lambda x: "Male" if x else "Female"
        )

    with c3:
        inputs["trestbps"] = st.number_input(
            "Resting BP (mm Hg)", 50, 300, int(df["trestbps"].median())
        )

    with c4:
        inputs["chol"] = st.number_input(
            "Cholesterol (mg/dl)", 100, 600, int(df["chol"].median())
        )

    st.divider()
    st.subheader("Cardiac Data")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        inputs["cp"] = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            index=safe_index([0, 1, 2, 3], int(df["cp"].median()))
        )

    with c6:
        inputs["thalach"] = st.number_input(
            "Max Heart Rate", 50, 250, int(df["thalach"].median())
        )

    with c7:
        inputs["exang"] = st.selectbox(
            "Exercise Angina",
            [0, 1],
            index=safe_index([0, 1], int(df["exang"].median())),
            format_func=lambda x: "Yes" if x else "No"
        )

    with c8:
        inputs["oldpeak"] = st.number_input(
            "ST Depression", 0.0, 10.0,
            float(df["oldpeak"].median()), step=0.1
        )

    st.divider()
    st.subheader("Medical History")

    c9, c10, c11, c12 = st.columns(4)

    with c9:
        inputs["fbs"] = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1],
            index=safe_index([0, 1], int(df["fbs"].median()))
        )

    with c10:
        inputs["restecg"] = st.selectbox(
            "Rest ECG",
            [0, 1, 2],
            index=safe_index([0, 1, 2], int(df["restecg"].median()))
        )

    with c11:
        inputs["slope"] = st.selectbox(
            "ST Slope",
            [0, 1, 2],
            index=safe_index([0, 1, 2], int(df["slope"].median()))
        )

    with c12:
        inputs["thal"] = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            index=safe_index([0, 1, 2, 3], int(df["thal"].median()))
        )

    inputs["ca"] = st.selectbox(
        "Major Vessels Colored",
        [0, 1, 2, 3],
        index=safe_index([0, 1, 2, 3], int(df["ca"].median()))
    )

    user_df = pd.DataFrame([inputs])
    return user_df[FEATURE_NAMES]


user_df = user_input_form(df)

# ============================ Prediction ============================
st.divider()

if st.button("Analyze Heart Disease Risk", use_container_width=True):
    st.session_state.analyzed = True

    try:
        scaled_input = scaler.transform(user_df)
        prediction = model.predict(scaled_input)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            st.error("High risk of heart disease detected")
        else:
            st.success("Low risk of heart disease detected")

        if probability is not None:
            st.metric("Risk Probability", f"{probability:.2%}")
            st.progress(probability)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ============================ Model Analytics ============================
if st.session_state.analyzed:
    with st.expander("Model Performance Analytics"):
        X = df[FEATURE_NAMES]
        y = df[TARGET_COL]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)

        tab1, tab2, tab3 = st.tabs(
            ["Confusion Matrix", "ROC Curve", "Feature Importance"]
        )

        with tab1:
            fig = plot_confusion_matrix(y_test, y_pred, model_choice)
            st.pyplot(fig, use_container_width=True)

        with tab2:
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
                fig = plot_roc_curve(y_test, y_probs, model_choice)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("ROC Curve not available for this model")

        with tab3:
            if model_choice == "Random Forest":
                fig = plot_feature_importance(
                    model.feature_importances_,
                    FEATURE_NAMES,
                    model_choice
                )
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Feature importance available only for Random Forest")

# ============================ Help Section ============================
st.divider()
st.subheader("Parameter Help Guide")

with st.expander("View detailed explanation of medical parameters"):
    st.markdown("""
    ### Chest Pain Type (cp)
    - **0**: Typical angina – chest pain related to reduced blood flow to the heart  
    - **1**: Atypical angina – chest pain not strongly linked to heart disease  
    - **2**: Non-anginal pain – pain unrelated to the heart  
    - **3**: Asymptomatic – no chest pain, but disease may still be present  

    ### Max Heart Rate Achieved (thalach)
    - Maximum heart rate recorded during exercise testing  
    - Lower values may indicate reduced heart performance  

    ### Exercise Induced Angina (exang)
    - **0**: No chest pain during exercise  
    - **1**: Chest pain occurs during physical activity  

    ### ST Depression (oldpeak)
    - Depression of ST segment during exercise  
    - Higher values indicate higher risk of heart disease  

    ### Fasting Blood Sugar > 120 mg/dl (fbs)
    - **0**: Blood sugar ≤ 120 mg/dl  
    - **1**: Blood sugar > 120 mg/dl  
    - High fasting sugar is linked to cardiovascular risk  

    ### Resting ECG Results (restecg)
    - **0**: Normal  
    - **1**: ST-T wave abnormality  
    - **2**: Left ventricular hypertrophy  

    ### ST Slope (slope)
    - **0**: Upsloping – generally lower risk  
    - **1**: Flat – moderate risk  
    - **2**: Downsloping – higher risk  

    ### Thalassemia (thal)
    - **0**: Normal  
    - **1**: Fixed defect – no blood flow improvement  
    - **2**: Reversible defect – blood flow improves with rest  
    - **3**: Unknown  
    """)
