from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class DrugDiscoveryModel:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.lr_model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        self.scaler = None
        
    def train_models(self, X_train, y_train):
        """
        Train both Random Forest and Logistic Regression models
        """
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        
        # Train Logistic Regression
        self.lr_model.fit(X_train, y_train)
        
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate both models and return metrics
        """
        metrics = {}
        
        # Random Forest predictions
        rf_pred = self.rf_model.predict(X_test)
        metrics['rf'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred),
            'recall': recall_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred)
        }
        
        # Logistic Regression predictions
        lr_pred = self.lr_model.predict(X_test)
        metrics['lr'] = {
            'accuracy': accuracy_score(y_test, lr_pred),
            'precision': precision_score(y_test, lr_pred),
            'recall': recall_score(y_test, lr_pred),
            'f1': f1_score(y_test, lr_pred)
        }
        
        return metrics, rf_pred
    
    def get_feature_importance(self, feature_names):
        """
        Get feature importance from Random Forest model
        """
        return self.rf_model.feature_importances_
    
    def predict_single(self, features):
        """
        Make prediction for a single molecule
        """
        rf_pred = self.rf_model.predict_proba(features)[0]
        lr_pred = self.lr_model.predict_proba(features)[0]
        
        return {
            'rf_prob': rf_pred[1],
            'lr_prob': lr_pred[1]
        }
