import streamlit as st
import pandas as pd
import joblib
import os
from src.visualize import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# -------------------- Configuration & Styling --------------------

st.set_page_config(
    page_title="HeartGuard AI | Predictive Analytics",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, modern look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        color: #2C3E50;
        font-weight: 700;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E1117; /* Dark Streamlit-like background */
        border-right: 1px solid #262730;
    }
    
    /* Sidebar Text */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #FAFAFA !important;
    }
    
    /* Card-like containers for inputs - Dark Mode */
    .stNumberInput, .stSelectbox {
        background-color: #262730; /* Dark grey */
        border-radius: 8px;
        color: white;
    }
    
    /* Input field text color override */
    div[data-baseweb="select"] > div, 
    input[type="number"] {
        color: white !important;
        background-color: #262730 !important;
    }
    
    label {
        color: #FAFAFA !important;
    }

    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #262730; /* Dark grey */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        border: 1px solid #444;
    }
    
    /* Info box styling (Sidebar insights) */
    .stAlert {
        background-color: #262730;
        color: white;
        border: 1px solid #444;
    }
    
    /* Highlight the prediction button */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #FF3333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Result container styling */
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    .high-risk {
        background-color: #fde8e8;
        border: 1px solid #fbd5d5;
        color: #9b1c1c;
    }
    .low-risk {
        background-color: #def7ec;
        border: 1px solid #bcf0da;
        color: #03543f;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Setup & Data Loading --------------------

# Constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'heart.csv')

@st.cache_resource
def load_resources():
    try:
        models = {
            "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')),
            "Random Forest": joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.pkl')),
            "SVM": joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
        }
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        st.stop()
        return None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"Dataset not found at {DATA_PATH}")
        st.stop()
        return None

models, scaler = load_resources()
df = load_dataset()
feature_names = df.drop("target", axis=1).columns.tolist()

# -------------------- Sidebar Controls --------------------

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

model_choice = st.sidebar.selectbox(
    "Select Prediction Model", 
    list(models.keys()),
    help="Choose the machine learning algorithm to perform the prediction."
)
model = models[model_choice]

st.sidebar.info(
    """
    **Model Insights:**
    - **Logistic Regression**: Linear classification, interpretable.
    - **Random Forest**: Ensemble method, high accuracy.
    - **SVM**: Effective in high-dimensional spaces.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 HeartGuard AI")

# -------------------- Main Interface --------------------

st.title("❤️ HeartGuard AI")
st.markdown("### Intelligent Heart Disease Risk Assessment")
st.markdown("Enter patient clinical data below to generate a real-time risk assessment using advanced machine learning models.")

st.divider()

# -------------------- User Inputs (Grouped) --------------------

def get_user_input(df):
    input_data = {}
    
    # --- Group 1: Personal & Vitals ---
    st.subheader("👤 Personal Details & Vitals")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        age_med = int(df['age'].median())
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=age_med, step=1)
        input_data['age'] = age
        
    with c2:
        sex_idx = int(df['sex'].median())
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female", index=sex_idx)
        input_data['sex'] = sex
        
    with c3:
        trestbps_med = int(df['trestbps'].median())
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=50, max_value=300, value=trestbps_med, help="Resting blood pressure on admission to the hospital")
        input_data['trestbps'] = trestbps
        
    with c4:
        chol_med = int(df['chol'].median())
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=chol_med)
        input_data['chol'] = chol

    # --- Group 2: Heart & Chest ---
    st.markdown("---")
    st.subheader("🫀 Cardiac Symptoms")
    c5, c6, c7, c8 = st.columns(4)
    
    with c5:
        cp_idx = int(df['cp'].median())
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], index=cp_idx, 
                          format_func=lambda x: f"Type {x}" if x==0 else f"Type {x}",
                          help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic")
        input_data['cp'] = cp
        
    with c6:
        thalach_med = int(df['thalach'].median())
        thalach = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=thalach_med)
        input_data['thalach'] = thalach
        
    with c7:
        exang_idx = int(df['exang'].median())
        exang = st.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=exang_idx)
        input_data['exang'] = exang
        
    with c8:
        oldpeak_med = float(df['oldpeak'].median())
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=oldpeak_med, step=0.1)
        input_data['oldpeak'] = oldpeak

    # --- Group 3: Medical/Other ---
    st.markdown("---")
    st.subheader("🧪 Medical History & Tests")
    c9, c10, c11, c12 = st.columns(4)
    
    with c9:
        fbs_idx = int(df['fbs'].median())
        fbs = st.selectbox("Fasting BS > 120", options=[0, 1], format_func=lambda x: "True" if x == 1 else "False", index=fbs_idx)
        input_data['fbs'] = fbs
        
    with c10:
        restecg_idx = int(df['restecg'].median())
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], index=restecg_idx, help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
        input_data['restecg'] = restecg
        
    with c11:
        slope_idx = int(df['slope'].median())
        slope = st.selectbox("ST Slope", options=[0, 1, 2], index=slope_idx)
        input_data['slope'] = slope
        
    with c12:
        thal_idx = int(df['thal'].median())
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], index=thal_idx)
        input_data['thal'] = thal
        
    # Extra field
    ca_idx = int(df['ca'].median())
    ca = st.selectbox("Major Vessels Colored (0-3)", options=[0, 1, 2, 3], index=ca_idx)
    input_data['ca'] = ca

    return pd.DataFrame([input_data])

user_input_df = get_user_input(df)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------- Prediction Logic --------------------

col_but1, col_but2, col_but3 = st.columns([1, 2, 1])
with col_but2:
    predict_btn = st.button("🚀 Analyze Risk Factor", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Processing clinical data..."):
        try:
            # Reorder columns to match training data
            user_input_df = user_input_df[feature_names]
            
            input_scaled = scaler.transform(user_input_df)
            prediction = model.predict(input_scaled)[0]
            
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_scaled)[0][1]
            else:
                prob = None

            # --- Result Display ---
            st.markdown("<br>", unsafe_allow_html=True)
            res_c1, res_c2 = st.columns(2)
            
            with res_c1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card high-risk">
                            <h3>⚠️ High Risk Detected</h3>
                            <p>The model predicts a high probability of heart disease.</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card low-risk">
                            <h3>✅ Low Risk Detected</h3>
                            <p>The model predicts a low probability of heart disease.</p>
                        </div>
                        """, unsafe_allow_html=True
                    )
            
            with res_c2:
                if prob is not None:
                    st.metric(
                        label="Risk Probability", 
                        value=f"{prob:.1%}", 
                        delta="High Condition" if prob > 0.5 else "Stable",
                        delta_color="inverse"
                    )
                    st.progress(prob)
                else:
                    st.info("Probability score not available for this model.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# -------------------- Analytics Section --------------------

st.markdown("---")
with st.expander("📊 View Model Performance Analytics", expanded=False):
    st.subheader(f"Model Diagnostics: {model_choice}")
    
    # Prepare test data (fixed split for consistency)
    from sklearn.model_selection import train_test_split
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Feature Importance"])
    
    with tab1:
        st.caption("Visualizes the performance of the classification model.")
        fig_cm = plot_confusion_matrix(y_test, y_pred, model_name=model_choice)
        st.pyplot(fig_cm)
        
    with tab2:
        st.caption("Illustrates the diagnostic ability of the binary classifier.")
        try:
            if hasattr(model, "predict_proba"):
                y_probs = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_probs = model.decision_function(X_test_scaled)
            
            fig_roc = plot_roc_curve(y_test, y_probs, model_name=model_choice)
            st.pyplot(fig_roc)
        except Exception as e:
            st.warning("ROC Curve not available.")
            
    with tab3:
        if model_choice == "Random Forest":
            st.caption("Shows which features most contributed to the model's decision.")
            fig_imp = plot_feature_importance(model.feature_importances_, feature_names, model_name=model_choice)
            st.pyplot(fig_imp)
        else:
            st.info("Feature Importance is only available for Random Forest models in this dashboard.")
