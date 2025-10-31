"""
Basic feed-forward neural network for tabular data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class TabularDataset(Dataset):
    """PyTorch dataset for tabular data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FeedForwardNN(nn.Module):
    """
    Basic feed-forward neural network with dropout for tabular data.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32], 
                 num_classes: int = 1, dropout: float = 0.3):
        super(FeedForwardNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        if num_classes == 1:
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class FeedForwardModel:
    """
    Wrapper class for training and using feed-forward neural networks.
    """
    
    def __init__(self, hidden_dims: list = [128, 64, 32], 
                 dropout: float = 0.3, learning_rate: float = 0.001,
                 device: Optional[str] = None):
        """
        Initialize the model.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_dim = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 50, batch_size: int = 256, 
              verbose: bool = True) -> dict:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        self.input_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train)) if len(np.unique(y_train)) > 2 else 1
        
        # Create model
        self.model = FeedForwardNN(
            self.input_dim, 
            self.hidden_dims, 
            num_classes=num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        if num_classes == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Data loaders
        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TabularDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        if num_classes == 1:
                            preds = (outputs > 0.5).cpu().numpy()
                        else:
                            preds = outputs.argmax(dim=1).cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_labels.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_acc = accuracy_score(all_labels, all_preds)
                val_f1 = f1_score(all_labels, all_preds, average='binary' if num_classes == 1 else 'macro')
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val F1: {val_f1:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        dataset = TabularDataset(X, np.zeros(len(X)))  # Dummy labels
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X).squeeze().cpu().numpy()
                predictions.extend(outputs)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        preds = self.predict(X)
        if preds.ndim == 1:
            # Binary classification
            return np.column_stack([1 - preds, preds])
        return preds


