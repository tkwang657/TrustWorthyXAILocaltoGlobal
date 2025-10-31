"""
XGBoost model wrapper for tabular data.
"""

import xgboost as xgb
import numpy as np
from typing import Optional, Dict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class XGBoostModel:
    """
    Wrapper for XGBoost model training and evaluation.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 objective: str = 'binary:logistic',
                 random_state: int = 42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            objective: Learning objective
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': objective,
            'random_state': random_state,
            'eval_metric': 'logloss' if 'binary' in objective else 'mlogloss'
        }
        self.model = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              early_stopping_rounds: int = 10, verbose: bool = True) -> Dict:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        if self.params['objective'].startswith('binary') or 'binary' in self.params['objective']:
            # Remove objective from params for classifier
            clf_params = {k: v for k, v in self.params.items() if k != 'objective'}
            clf_params['objective'] = self.params['objective']
            clf_params['eval_metric'] = 'logloss'
            self.model = xgb.XGBClassifier(**clf_params)
        else:
            reg_params = {k: v for k, v in self.params.items() if k != 'objective'}
            self.model = xgb.XGBRegressor(**reg_params)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose=verbose
        )
        
        history = {}
        if hasattr(self.model, 'evals_result_') and self.model.evals_result_:
            history = self.model.evals_result_
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importances.
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
            
        Returns:
            Dictionary mapping feature names to importances
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importances = self.model.get_booster().get_score(importance_type=importance_type)
        return importances

