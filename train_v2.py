import os
import logging
from src.preprocess import load_data, preprocess_data
from src.model import (
    train_logistic_regression, 
    train_random_forest, 
    train_svm, 
    train_ensemble_model, 
    save_model
)
from src.evaluate import evaluate_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load and Preprocess Data
    data_path = 'data/heart.csv'
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    logger.info("Loading dataset...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # 2. Train Individual Models (Fine-tuned & Recall Optimized)
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    svm_model = train_svm(X_train, y_train)

    # 3. Train Ensemble Model
    ensemble_model = train_ensemble_model(lr_model, rf_model, svm_model, X_train, y_train)

    # 4. Evaluate Models
    logger.info("--- EVALUATION: Logistic Regression ---")
    evaluate_model(lr_model, X_test, y_test)
    
    logger.info("--- EVALUATION: Random Forest ---")
    evaluate_model(rf_model, X_test, y_test)
    
    logger.info("--- EVALUATION: SVM ---")
    evaluate_model(svm_model, X_test, y_test)
    
    logger.info("--- EVALUATION: Ensemble (Consensus) ---")
    evaluate_model(ensemble_model, X_test, y_test)

    # 5. Save Everything
    save_model(lr_model, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    save_model(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
    save_model(svm_model, os.path.join(models_dir, 'svm_model.pkl'))
    save_model(ensemble_model, os.path.join(models_dir, 'ensemble_model.pkl'))
    save_model(scaler, os.path.join(models_dir, 'scaler.pkl'))

    logger.info("All models leveled up and saved successfully!")

if __name__ == "__main__":
    main()
