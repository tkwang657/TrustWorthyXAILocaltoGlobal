"""
Basic feed-forward neural network for tabular data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight



class MortageDataset(Dataset):
    
    def __init__(self, X, y):

        X_values = np.array(X)
        y_values = np.array(y)

        self.X = torch.tensor(X_values, dtype=torch.float32)
        self.y = torch.tensor(y_values, dtype=torch.long)# so that its 0 to 7 and not 1 to 8
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




class FeedForwardNN(nn.Module):
    """
    Deep feed-forward neural network with dropout for tabular data.
    """

    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: list = [512, 512, 256, 256, 128, 128, 64], 
                 num_classes: int = 1, 
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes if num_classes > 1 else 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)




class FeedForwardModel:
    """
    Wrapper class for training and using feed-forward neural networks.
    """
    
    def __init__(self,
                 hidden_dims: list = [128, 64, 32], 
                 dropout: float = 0.3, 
                 learning_rate: float = 0.0005,
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
        self.num_classes = None
        
    def train(self, 
              X_train, 
              y_train, 
              X_val=None, 
              y_val=None,
              epochs: int = 50, 
              batch_size: int = 256) -> dict:
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

        # Data loaders
        train_dataset = MortageDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        if X_val is not None and y_val is not None:
            val_dataset = MortageDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.input_dim = train_dataset.X.shape[1]

        # Determine number of classes
        unique_classes = np.unique(train_dataset.y)
        self.num_classes = len(unique_classes)
        
        # Create model
        self.model = FeedForwardNN(
            self.input_dim, 
            self.hidden_dims, 
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        # weighting loss since theres WAY more of some classes than others in the dataset
        classes = np.unique(train_dataset.y)  # keep 1–8
        print("CLASSES:", classes)
        print(train_dataset.y.unique())
        class_weights = compute_class_weight('balanced', classes=classes, y=train_dataset.y.cpu().numpy())
        # Map class weights into tensor in the correct order
        weight_tensor = torch.zeros(len(classes), dtype=torch.float32)
        for i, cls in enumerate(classes):
            weight_tensor[i] = class_weights[i]
        weight_tensor = weight_tensor.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        epoch_iterator = tqdm(range(epochs), desc="Training", unit="epoch") # Create progress bar
        
        for epoch in epoch_iterator:

            # Training
            self.model.train()
            train_loss = 0.0
            valid_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()

                optimizer.step()
                
                train_loss += loss.item()
                valid_batches += 1
            
            # Calculate average loss only over valid batches
            train_loss /= valid_batches

            history['train_loss'].append(train_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                
                self.model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []
                valid_val_batches = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        valid_val_batches += 1
                        
                        preds = outputs.argmax(dim=1).cpu().numpy()
                        
                        all_preds.extend(preds)
                        all_labels.extend(batch_y.cpu().numpy())
                
                # Calculate average loss only over valid batches
                val_loss /= valid_val_batches
            
                val_acc = accuracy_score(all_labels, all_preds)
                val_f1 = f1_score(all_labels, all_preds, average='macro')
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                
                # Update epoch progress bar
                try:
                    epoch_iterator.set_postfix({
                        'train_loss': f'{train_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'val_acc': f'{val_acc:.4f}',
                        'val_f1': f'{val_f1:.4f}'
                    })
                except:
                    # Fallback to print if tqdm update fails
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                            f"Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f} | "
                            f"Val Acc: {val_acc:.4f} | "
                            f"Val F1: {val_f1:.4f}")
            else:
                # No validation set
                try:
                    epoch_iterator.set_postfix({'train_loss': f'{train_loss:.4f}'})
                except:
                    print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        # Use dummy labels - type doesn't matter for prediction
        dummy_y = np.zeros(len(X), dtype=np.int64)
        dataset = MortageDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                # Output is (batch_size, num_classes), apply softmax and get argmax
                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        # Use dummy labels - type doesn't matter for prediction
        dummy_y = np.zeros(len(X), dtype=np.int64)
        dataset = MortageDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                # Output is (batch_size, num_classes), apply softmax
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities.extend(probs)
        
        return np.array(probabilities)


