import logging
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model performance...")
    print("Evaluating model performance...")
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    logger.info(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    print(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")
    print(f"Classification Report:\n{report}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        logger.info(f"ROC AUC: {roc_auc:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    import os
    import joblib
    from src.preprocess import load_data, preprocess_data
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart.csv')
    
    if not os.path.exists(MODEL_DIR):
        print("Error: models directory not found.")
    else:
        # Load data
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        X_test_scaled = scaler.transform(X_test)
        
        # Load and Evaluate each model
        model_files = {
            "Logistic Regression": "logistic_regression_model.pkl",
            "Random Forest": "random_forest_model.pkl",
            "SVM": "svm_model.pkl",
            "Ensemble": "ensemble_model.pkl"
        }
        
        for name, filename in model_files.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                print(f"\n--- {name} ---")
                model = joblib.load(path)
                evaluate_model(model, X_test_scaled, y_test)
            else:
                print(f"Skipping {name}: {filename} not found.")
