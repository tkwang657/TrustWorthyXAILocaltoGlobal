"""
TabNet model wrapper for tabular data.

TabNet uses sequential attention to select features at each decision step,
providing both strong performance and some interpretability.
"""

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from typing import Optional, Dict
from sklearn.metrics import accuracy_score, f1_score


class TabNetModel:
    """
    Wrapper for TabNet model training and evaluation.
    """
    
    def __init__(self,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 gamma: float = 1.3,
                 lambda_sparse: float = 1e-3,
                 optimizer_fn: torch.optim.Optimizer = torch.optim.Adam,
                 optimizer_params: Dict = {'lr': 2e-2},
                 scheduler_fn: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 scheduler_params: Dict = {},
                 mask_type: str = 'sparsemax',
                 seed: int = 42,
                 device_name: str = 'auto'):
        """
        Initialize TabNet model.
        
        Args:
            n_d: Dimension of the decision embedding
            n_a: Dimension of the attention embedding
            n_steps: Number of steps in the encoder
            gamma: Coefficient for feature reusage
            lambda_sparse: Sparsity regularization
            optimizer_fn: Optimizer class
            optimizer_params: Optimizer parameters
            scheduler_fn: Learning rate scheduler (optional)
            scheduler_params: Scheduler parameters
            mask_type: 'sparsemax' or 'entmax'
            seed: Random seed
            device_name: 'auto', 'cpu', or 'cuda'
        """
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.seed = seed
        self.device_name = device_name
        self.model = None
        self.is_classification = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None,
              task: str = 'classification',
              max_epochs: int = 100,
              batch_size: int = 1024,
              virtual_batch_size: int = 128,
              patience: int = 15,
              verbose: bool = True) -> Dict:
        """
        Train TabNet model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            task: 'classification' or 'regression'
            max_epochs: Maximum number of epochs
            batch_size: Batch size
            virtual_batch_size: Virtual batch size for ghost batch norm
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        self.is_classification = (task == 'classification')
        
        # Create model
        if self.is_classification:
            num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else 2
            self.model = TabNetClassifier(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                optimizer_fn=self.optimizer_fn,
                optimizer_params=self.optimizer_params,
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                mask_type=self.mask_type,
                seed=self.seed,
                device_name=self.device_name
            )
        else:
            self.model = TabNetRegressor(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                optimizer_fn=self.optimizer_fn,
                optimizer_params=self.optimizer_params,
                scheduler_fn=self.scheduler_fn,
                scheduler_params=self.scheduler_params,
                mask_type=self.mask_type,
                seed=self.seed,
                device_name=self.device_name
            )
        
        # Prepare data (ensure numpy arrays)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        eval_set = None
        eval_name = None
        if X_val is not None and y_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
            eval_set = [(X_val, y_val)]
            eval_name = ['valid']
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            max_epochs=max_epochs,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            patience=patience,
            verbose=verbose
        )
        
        history = {
            'train_loss': self.model.history['loss'] if 'loss' in self.model.history else [],
            'val_loss': self.model.history['val_loss'] if 'val_loss' in self.model.history else []
        }
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(np.array(X))
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if not self.is_classification:
            raise ValueError("predict_proba only available for classification tasks")
        return self.model.predict_proba(np.array(X))
    
    def explain(self, X, normalize=True):
        """
        Get feature importance masks for local explanations.
        
        TabNet provides attention masks showing which features
        were used at each decision step.
        
        Args:
            X: Input data
            normalize: Whether to normalize masks
            
        Returns:
            Feature masks showing attention weights
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.explain(np.array(X), normalize=normalize)


