import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_logistic_regression(X_train , Y_train):
    logger.info("Starting Logistic Regression fine-tuning (Recall-optimized)...")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100], 
        'solver': ['lbfgs', 'liblinear'],
        'class_weight': ['balanced', None]
    }
    # Using recall scoring because missing a sick patient is high risk
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='recall', n_jobs=-1)
    grid.fit(X_train, Y_train)
    logger.info(f"LR Best Params: {grid.best_params_}")
    return grid.best_estimator_

def train_random_forest(X_train , Y_train):
    logger.info("Starting Random Forest fine-tuning (Recall-optimized)...")
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, 
                              n_iter=15, cv=5, scoring='recall', n_jobs=-1, random_state=42)
    rand.fit(X_train, Y_train)
    logger.info(f"RF Best Params: {rand.best_params_}")
    return rand.best_estimator_

def train_svm(X_train , Y_train):
    logger.info("Starting SVM fine-tuning (Recall-optimized)...")
    param_grid = {
        'C': [0.1, 1, 10, 100], 
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='recall', n_jobs=-1)
    grid.fit(X_train, Y_train)
    logger.info(f"SVM Best Params: {grid.best_params_}")
    return grid.best_estimator_

def train_ensemble_model(lr_model, rf_model, svm_model, X_train, Y_train):
    logger.info("Building Ensemble Voting Classifier...")
    # Combine the best of all worlds
    ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('rf', rf_model),
            ('svm', svm_model)
        ],
        voting='soft' # Uses probabilities for a "consensus" prediction
    )
    ensemble.fit(X_train, Y_train)
    logger.info("Ensemble model training complete.")
    return ensemble

def save_model(model , file_path):
    logger.info(f"Saving model to {file_path}...")
    joblib.dump(model , file_path)
    logger.info("Model saved successfully.")

