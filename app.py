# ==================== Imports ====================
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from src.visualize import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)

# ==================== Page Config ====================
st.set_page_config(
    page_title="HeartGuard AI | Predictive Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Paths ====================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "heart.csv"

# ==================== Resource Loading ====================
@st.cache_resource
def load_models_and_scaler():
    models = {
        "Logistic Regression": MODEL_DIR / "logistic_regression_model.pkl",
        "Random Forest": MODEL_DIR / "random_forest_model.pkl",
        "SVM": MODEL_DIR / "svm_model.pkl",
    }

    loaded_models = {}
    for name, path in models.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path.name}")
        loaded_models[name] = joblib.load(path)

    scaler_path = MODEL_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError("Missing scaler.pkl")

    scaler = joblib.load(scaler_path)
    return loaded_models, scaler


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError("heart.csv not found")
    return pd.read_csv(DATA_PATH)


# ==================== Safe Load ====================
try:
    models, scaler = load_models_and_scaler()
    df = load_dataset()
except Exception as e:
    st.error(f"Startup Error: {e}")
    st.stop()

FEATURE_NAMES = df.drop(columns="target").columns.tolist()

# ==================== Sidebar ====================
st.sidebar.title("Configuration")
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    list(models.keys())
)
model = models[model_choice]

# ==================== Input Builder ====================
def safe_index(options, value):
    return options.index(value) if value in options else 0


def user_input_form(dataframe):
    st.subheader("👤 Patient Information")
    inputs = {}

    cols = st.columns(4)
    with cols[0]:
        inputs["age"] = st.number_input(
            "Age",
            1, 120,
            int(dataframe["age"].median())
        )

    with cols[1]:
        sex_val = int(dataframe["sex"].median())
        options = [0, 1]
        inputs["sex"] = st.selectbox(
            "Sex",
            options,
            index=safe_index(options, sex_val),
            format_func=lambda x: "Male" if x else "Female"
        )

    with cols[2]:
        inputs["trestbps"] = st.number_input(
            "Resting BP",
            50, 300,
            int(dataframe["trestbps"].median())
        )

    with cols[3]:
        inputs["chol"] = st.number_input(
            "Cholesterol",
            100, 600,
            int(dataframe["chol"].median())
        )

    st.divider()
    st.subheader("🫀 Cardiac Data")

    cols = st.columns(4)
    with cols[0]:
        inputs["cp"] = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3],
            index=safe_index([0, 1, 2, 3], int(dataframe["cp"].median()))
        )

    with cols[1]:
        inputs["thalach"] = st.number_input(
            "Max Heart Rate",
            50, 250,
            int(dataframe["thalach"].median())
        )

    with cols[2]:
        inputs["exang"] = st.selectbox(
            "Exercise Angina",
            [0, 1],
            index=safe_index([0, 1], int(dataframe["exang"].median())),
            format_func=lambda x: "Yes" if x else "No"
        )

    with cols[3]:
        inputs["oldpeak"] = st.number_input(
            "ST Depression",
            0.0, 10.0,
            float(dataframe["oldpeak"].median()),
            step=0.1
        )

    st.divider()
    st.subheader("🧪 Medical History")

    cols = st.columns(4)
    with cols[0]:
        inputs["fbs"] = st.selectbox(
            "Fasting BS > 120",
            [0, 1],
            index=safe_index([0, 1], int(dataframe["fbs"].median()))
        )

    with cols[1]:
        inputs["restecg"] = st.selectbox(
            "Rest ECG",
            [0, 1, 2],
            index=safe_index([0, 1, 2], int(dataframe["restecg"].median()))
        )

    with cols[2]:
        inputs["slope"] = st.selectbox(
            "ST Slope",
            [0, 1, 2],
            index=safe_index([0, 1, 2], int(dataframe["slope"].median()))
        )

    with cols[3]:
        inputs["thal"] = st.selectbox(
            "Thalassemia",
            [0, 1, 2, 3],
            index=safe_index([0, 1, 2, 3], int(dataframe["thal"].median()))
        )

    inputs["ca"] = st.selectbox(
        "Major Vessels Colored",
        [0, 1, 2, 3],
        index=safe_index([0, 1, 2, 3], int(dataframe["ca"].median()))
    )

    df_input = pd.DataFrame([inputs])
    return df_input[FEATURE_NAMES]


user_df = user_input_form(df)

# ==================== Prediction ====================
st.divider()
if st.button("🚀 Analyze Risk", use_container_width=True):
    try:
        scaled = scaler.transform(user_df.values)
        pred = model.predict(scaled)[0]

        prob = (
            model.predict_proba(scaled)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        if pred:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk Detected")

        if prob is not None:
            st.metric("Risk Probability", f"{prob:.2%}")
            st.progress(prob)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==================== Analytics ====================
with st.expander("📊 Model Analytics"):
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_NAMES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_test_scaled = scaler.transform(X_test.values)
    y_pred = model.predict(X_test_scaled)

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC", "Importance"])

    with tab1:
        st.pyplot(plot_confusion_matrix(y_test, y_pred, model_choice))

    with tab2:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_scaled)[:, 1]
            st.pyplot(plot_roc_curve(y_test, probs, model_choice))
        else:
            st.info("ROC not supported for this model")

    with tab3:
        if model_choice == "Random Forest":
            st.pyplot(
                plot_feature_importance(
                    model.feature_importances_,
                    FEATURE_NAMES,
                    model_choice
                )
            )
        else:
            st.info("Feature importance only for Random Forest")
