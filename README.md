# Heart Disease Prediction Multi-Model Dashboard

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project is an interactive **Heart Disease Prediction Dashboard** powered by Machine Learning. It allows users to input medical data (such as age, cholesterol levels, chest pain type, etc.) and receive a real-time risk assessment for heart disease.

The system utilizes three powerful classification algorithms to ensure reliable predictions:
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

Built with **Streamlit**, the application provides a user-friendly interface for both patients and healthcare professionals to explore risk factors and model performance metrics.

## 🚀 Key Features

- **Interactive Risk Prediction**: Easy-to-use side panel and form to input patient health metrics.
- **Multi-Model Support**: Switch between Logistic Regression, Random Forest, and SVM to compare predictions.
- **Real-time Probability**: Displays not just the classification (High/Low Risk) but also the confidence probability.
- **Visual Analytics**:
    - **Confusion Matrix**: To visualize model accuracy on test data.
    - **ROC Curve**: To analyze the trade-off between sensitivity and specificity.
    - **Feature Importance**: (Random Forest only) To understand which health factors contribute most to the prediction.
- **Responsive Design**: Clean and professional UI layout.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: Scikit-Learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Backend Model Loading**: Joblib

## 📂 Project Structure

```
HeartDiseasePredictionMultiModel/
├── app.py                # Main Streamlit application entry point
├── requirements.txt      # Project dependencies
├── models/               # Pre-trained ML models (.pkl files)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── scaler.pkl        # Data scaler for normalization
├── data/                 # Dataset directory
│   └── heart.csv         # Source dataset (Cleveland Heart Disease Data)
└── src/                  # Helper modules
    ├── visualize.py      # Plotting functions (ROC, Confusion Matrix)
    ├── evaluate.py       # Metrics evaluation
    └── preprocess.py     # Data cleaning pipelines
```

## ⚙️ Setup and Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/HeartDiseasePredictionMultiModel.git
cd HeartDiseasePredictionMultiModel
```

### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ How to Run

To launch the dashboard, use the `streamlit run` command:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

## 📊 Dataset Details

The model is trained on the **Cleveland Heart Disease Dataset** from the UCI Machine Learning Repository. It contains 14 key attributes:
- **Age, Sex**
- **CP**: Chest pain type (4 values)
- **Trestbps**: Resting blood pressure
- **Chol**: Serum cholestoral in mg/dl
- **Fbs**: Fasting blood sugar > 120 mg/dl
- **Restecg**: Resting electrocardiographic results (values 0,1,2)
- **Thalach**: Maximum heart rate achieved
- **Exang**: Exercise induced angina
- **Oldpeak**: ST depression induced by exercise relative to rest
- **Slope**: The slope of the peak exercise ST segment
- **Ca**: Number of major vessels (0-3) colored by flourosopy
- **Thal**: Thalassemia status

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is Developed by Muhammad Hanzala
