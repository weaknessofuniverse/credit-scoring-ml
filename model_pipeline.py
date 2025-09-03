import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class CreditScoringPipeline:
    
    def __init__(self):
        self.models = {}
        self.feature_names = None
        
    def get_available_models(self):
        """Return a list of available models with their parameter grids"""
        return {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced'),
                'param_grid': {
                    'logisticregression__C': [0.1, 1, 10],
                    'logisticregression__solver': ['liblinear', 'saga'],
                    'logisticregression__penalty': ['l1', 'l2']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'param_grid': {
                    'randomforestclassifier__n_estimators': [100, 200],
                    'randomforestclassifier__max_depth': [10, 20, None],
                    'randomforestclassifier__min_samples_split': [2, 5]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'param_grid': {
                    'gradientboostingclassifier__n_estimators': [100, 200],
                    'gradientboostingclassifier__learning_rate': [0.05, 0.1],
                    'gradientboostingclassifier__max_depth': [3, 5]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42, class_weight='balanced'),
                'param_grid': {
                    'svc__C': [0.1, 1, 10],
                    'svc__kernel': ['linear', 'rbf'],
                    'svc__gamma': ['scale', 'auto']
                }
            },
            'Bagging': {
                'model': BaggingClassifier(random_state=42),
                'param_grid': {
                    'baggingclassifier__n_estimators': [10, 20],
                    'baggingclassifier__max_samples': [0.5, 0.8, 1.0],
                    'baggingclassifier__max_features': [0.5, 0.8, 1.0]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'param_grid': {
                    'kneighborsclassifier__n_neighbors': [3, 5, 7, 9],
                    'kneighborsclassifier__weights': ['uniform', 'distance'],
                    'kneighborsclassifier__p': [1, 2]
                }
            }
        }
    
    def create_pipeline(self, model, use_smote=True, scale_features=False):
        """Create a pipeline with optional SMOTE and feature scaling"""
        steps = []
        
        if scale_features:
            steps.append(('scaler', StandardScaler()))
            
        if use_smote:
            steps.append(('smote', SMOTE(random_state=42)))
            
        # Add the model with its name in lowercase
        model_name = type(model).__name__.lower()
        steps.append((model_name, model))
        
        return Pipeline(steps)
    
    def train_model(self, model_name, X_train, y_train, X_val, y_val, 
                   use_smote=True, scale_features=False, cv=3, n_jobs=-1, 
                   search_method='grid'):
        """Train and evaluate a single model"""
        
        available_models = self.get_available_models()
        
        print(f"Training {model_name}...")
        
        # Get model and parameter grid
        model_info = available_models[model_name]
        model = model_info['model']
        param_grid = model_info['param_grid']
        
        # Create pipeline
        pipeline = self.create_pipeline(model, use_smote, scale_features)
        
        # Choose search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=10,  # Number of parameter settings sampled
                cv=cv,
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=1,
                random_state=42
            )
        
        # Train the model with hyperparameter tuning
        search.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_val_pred_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        
        # Store results
        self.models[model_name] = {
            'best_estimator': search.best_estimator_,
            'best_params': search.best_params_,
            'cv_score': search.best_score_,
            'val_score': val_auc,
            'search': search
        }
        
        print(f"Best parameters: {search.best_params_}")
        print(f"CV AUC: {search.best_score_:.4f}")
        print(f"Validation AUC: {val_auc:.4f}")
        
        return search.best_estimator_
    
    def feature_importance(self, model_name, top_n=15):
        """Display feature importance for tree-based models"""
        
        pipeline = self.models[model_name]['best_estimator']
        model = pipeline.steps[-1][1]  # Get the actual model from the pipeline
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
            
            if self.feature_names is None:
                self.feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df)
            plt.title(f'Top {top_n} Feature Importances - {model_name}')
            plt.tight_layout()
            plt.show()
            
            return feature_importance_df
        
        else:
            print(f"Feature importance is not available for {model_name}.")
            return None
    
    def save_model(self, model, path):
        """Save a model to disk"""
        
        joblib.dump(model, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path):
        """Load a saved model from disk"""
        
        model = joblib.load(path)
        print(f"Model loaded from: {path}")
        return model