import logging
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model performance...")
    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{report}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        logger.info(f"ROC AUC: {roc_auc:.4f}")