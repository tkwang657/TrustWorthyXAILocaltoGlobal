"""
Example usage script demonstrating the pipeline components.

This script shows how to use individual components without running
the full pipeline.
"""


# ACTION_TAKEN_LABELS = {
#     1: "Loan originated",
#     2: "Application approved but not accepted",
#     3: "Application denied",
#     4: "Application withdrawn by applicant",
#     5: "File closed for incompleteness",
#     6: "Purchased loan",
#     7: "Preapproval request denied",
#     8: "Preapproval request approved but not accepted"
# }

import numpy as np
from models.feedforward import FeedForwardModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def data_loading():
    """Load and preprocess data."""
    print("="*60)
    print("Example 1: Data Loading")
    print("="*60)
    
    # Example CSV: "data/my_data.csv"
    df = pd.read_csv("data/my_data.csv")

    # Features and target
    target_col = "action_taken"  # change this to your column
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def model_training(X_train, y_train, X_test, y_test):
    """=Train a model."""
    print("\n" + "="*60)
    print("Example 2: Model Training")
    print("="*60)
    
    # Train 
    print("Training XGBoost...")
    ff_model = FeedForwardModel()
    ff_model.train(X_train, y_train, X_test, y_test, verbose=False)
    
    # Evaluate
    y_pred = ff_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"FFNN accuracy: {acc:.4f}")
    
    return ff_model


if __name__ == '__main__':
    print("Local-to-Global Explanations - Example Usage\n")

    try:
        X_train, X_test, y_train, y_test = data_loading()
        
        model = model_training(X_train, y_train, X_test, y_test)
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        
    except FileNotFoundError:
        print("\nError: HMDA data not found.")
        print("Please download data from:")
        print("https://ffiec.cfpb.gov/data-publication/dynamic-national-loan-level-dataset/2024")
        print("And place CSV files in data/raw/")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


