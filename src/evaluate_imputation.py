"""
Evaluate imputation performance across all target variables.
This script compares model performance with and without the improved imputation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from data_loader import load_raw_data, preprocess_data
from config import TARGET_COLUMNS_DICT

def evaluate_target(score_type, task='classification'):
    """
    Evaluate imputation performance on a specific target variable.
    
    Parameters:
    -----------
    score_type : str
        The frailty score type ('FRIED' or 'FRAGIRE18')
    task : str
        The task type ('classification' only - regression removed)
    
    Returns:
    --------
    scores : np.array
        Cross-validation scores (ROC AUC for classification)
    """
    # Load raw data with missing values
    raw_data = load_raw_data()
    
    # Get target column
    target_col = TARGET_COLUMNS_DICT[score_type][task]
    
    # Mask for rows with non-missing target
    valid_mask = ~raw_data[target_col].isna()
    
    # Get data with preprocessing (including imputation)
    X, y = preprocess_data(raw_data, score_type, task)
    
    # Only keep rows with non-missing targets in the original data
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        if task == 'classification':
            model = lgb.LGBMClassifier(
                objective='binary', 
                random_state=42,
                n_estimators=100,
                is_unbalance=True
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        if task == 'classification':
            y_pred = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred)
        
        scores.append(score)
    
    return np.array(scores)

def main():
    """
    Main function to evaluate imputation performance.
    
    For each target variable:
    - FRIED (classification only)
    - FRAGIRE18 (classification only)
    
    We train models on the imputed data and report cross-validation scores.
    """
    results = {}
    
    # Classification tasks
    results['FRIED_classification'] = evaluate_target('FRIED', 'classification')
    results['FRAGIRE18_classification'] = evaluate_target('FRAGIRE18', 'classification')
    
    print("\nImputation Evaluation Results:")
    print("-------------------------------")
    print("Classification (ROC AUC, higher is better):")
    print(f"  FRIED: {results['FRIED_classification'].mean():.3f} ± {results['FRIED_classification'].std():.3f}")
    print(f"  FRAGIRE18: {results['FRAGIRE18_classification'].mean():.3f} ± {results['FRAGIRE18_classification'].std():.3f}")

if __name__ == "__main__":
    main()
