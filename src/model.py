import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_logistic_regression(X_train , Y_train):
    from sklearn.model_selection import GridSearchCV
    logger.info("Starting Logistic Regression training with GridSearchCV...")
    param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
    grid = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)
    logger.info(f"Logistic Regression training complete. Best parameters: {grid.best_params_}")
    return grid.best_estimator_

def train_random_forest(X_train , Y_train):
    from sklearn.model_selection import RandomizedSearchCV
    logger.info("Starting Random Forest training with RandomizedSearchCV...")
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1)
    rand.fit(X_train, Y_train)
    logger.info(f"Random Forest training complete. Best parameters: {rand.best_params_}")
    return rand.best_estimator_

def train_svm(X_train , Y_train):
    from sklearn.model_selection import GridSearchCV
    logger.info("Starting SVM training with GridSearchCV...")
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train, Y_train)
    logger.info(f"SVM training complete. Best parameters: {grid.best_params_}")
    return grid.best_estimator_

def save_model(model , file_path):
    logger.info(f"Saving model to {file_path}...")
    joblib.dump(model , file_path)
    logger.info("Model saved successfully.")

