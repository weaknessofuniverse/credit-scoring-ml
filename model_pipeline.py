import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects specific features from dataframe"""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]

class CreditScoringPipeline:
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def define_models(self, use_smote=True):
        """Define models and their parameter grids"""
        
        # Base models
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'Bagging': BaggingClassifier(random_state=42)
        }
        
        # Parameter grids for tuning
        param_grids = {
            'LogisticRegression': {
                'model__C': [0.1, 1, 10],
                'model__solver': ['liblinear', 'saga']
            },
            'RandomForest': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'model__n_estimators': [100, 200],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5]
            },
            'SVM': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf']
            },
            'Bagging': {
                'model__n_estimators': [10, 20],
                'model__max_samples': [0.5, 0.8, 1.0],
                'model__max_features': [0.5, 0.8, 1.0]
            }
        }
        
        # Create pipelines
        for name, model in models.items():
            if use_smote:
                # Pipeline with SMOTE for handling class imbalance
                pipeline = make_imb_pipeline(
                    SMOTE(random_state=42),
                    model
                )
            else:
                pipeline = Pipeline([('model', model)])
                
            self.models[name] = {
                'pipeline': pipeline,
                'param_grid': param_grids.get(name, {})
            }
    
    def train_models(self, X_train, y_train, cv=3, n_jobs=-1):
        """Train all models with hyperparameter tuning"""
        
        print("Training models with hyperparameter tuning...")
        
        for name, model_info in self.models.items():
            print(f"\nTraining {name}...")
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model_info['pipeline'],
                param_grid=model_info['param_grid'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.models[name]['best_estimator'] = grid_search.best_estimator_
            self.models[name]['best_params'] = grid_search.best_params_
            self.models[name]['best_score'] = grid_search.best_score_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    
    def evaluate_models(self, X_val, y_val):
        """Evaluate all models on validation set"""
        
        print("\nEvaluating models on validation set...")
        
        for name, model_info in self.models.items():
            # Predict probabilities
            y_pred_proba = model_info['best_estimator'].predict_proba(X_val)[:, 1]
            
            # Calculate AUC
            auc_score = roc_auc_score(y_val, y_pred_proba)
            self.models[name]['val_auc'] = auc_score
            
            print(f"{name}: Validation AUC = {auc_score:.4f}")
    
    def select_best_model(self):
        """Select the best performing model based on validation AUC"""
        
        best_auc = -1
        for name, model_info in self.models.items():
            if model_info['val_auc'] > best_auc:
                best_auc = model_info['val_auc']
                self.best_model = model_info['best_estimator']
                self.best_model_name = name
                
        print(f"\nBest model: {self.best_model_name} with AUC: {best_auc:.4f}")
    
    def feature_importance(self, feature_names):
        """Display feature importance for tree-based models"""
        
        if hasattr(self.best_model.steps[-1][-1], 'feature_importances_'):
            # For tree-based models
            importances = self.best_model.steps[-1][-1].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title("Feature Importances")
            plt.bar(range(min(20, len(importances))), 
                   importances[indices][:20],
                   color="r", align="center")
            plt.xticks(range(min(20, len(importances))), 
                      [feature_names[i] for i in indices[:20]], rotation=90)
            plt.xlim([-1, min(20, len(importances))])
            plt.tight_layout()
            plt.show()
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return feature_importance_df
        
        else:
            print("Feature importance is not available for this model type.")
            return None
    
    def final_training(self, X, y):
        """Train the best model on the entire dataset"""
        
        print(f"\nTraining final {self.best_model_name} model on full dataset...")
        self.final_model = self.best_model.fit(X, y)
    
    def predict(self, X):
        """Make predictions using the final model"""
        
        return self.final_model.predict_proba(X)[:, 1]
    
    def save_model(self, path):
        """Save the final model to disk"""
        
        joblib.dump(self.final_model, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load a saved model from disk"""
        
        self.final_model = joblib.load(path)
        print(f"Model loaded from: {path}")