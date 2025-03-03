"""
Evaluate imputation performance across all target variables.
This script compares model performance with and without the improved imputation.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
import lightgbm as lgb
from data_loader import load_data
import config

def evaluate_target(score_type, task, n_splits=5):
    """
    Evaluate model performance for a specific target variable.
    Uses cross-validation to get robust performance estimates.
    """
    print(f"\nEvaluating {score_type} ({task})...")
    
    # Load data
    X, y, feature_names = load_data(score_type, task)
    
    # Initialize model (using LightGBM for quick evaluation)
    if task == 'classification':
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        scoring = 'roc_auc'
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        scoring = 'neg_root_mean_squared_error'
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    
    # Print results
    if task == 'classification':
        print(f"ROC AUC scores: {scores}")
        print(f"Mean ROC AUC: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    else:
        rmse_scores = -scores  # Convert negative RMSE back to positive
        print(f"RMSE scores: {rmse_scores}")
        print(f"Mean RMSE: {rmse_scores.mean():.3f} (+/- {rmse_scores.std() * 2:.3f})")
    
    return scores

def main():
    """
    Evaluate all target variables:
    - FRIED (classification and regression)
    - FRAGIRE18 (classification and regression)
    """
    results = {}
    
    # Classification tasks
    results['FRIED_classification'] = evaluate_target('FRIED', 'classification')
    results['FRAGIRE18_classification'] = evaluate_target('FRAGIRE18', 'classification')
    
    # Regression tasks
    results['FRIED_regression'] = evaluate_target('FRIED', 'regression')
    results['FRAGIRE18_regression'] = evaluate_target('FRAGIRE18', 'regression')
    
    # Save results
    results_df = pd.DataFrame({
        'Target': ['FRIED_cls', 'FRAGIRE18_cls', 'FRIED_reg', 'FRAGIRE18_reg'],
        'Mean Score': [
            results['FRIED_classification'].mean(),
            results['FRAGIRE18_classification'].mean(),
            -results['FRIED_regression'].mean(),  # Convert back to positive RMSE
            -results['FRAGIRE18_regression'].mean()
        ],
        'Std Dev': [
            results['FRIED_classification'].std(),
            results['FRAGIRE18_classification'].std(),
            results['FRIED_regression'].std(),
            results['FRAGIRE18_regression'].std()
        ]
    })
    
    print("\nSummary of Results:")
    print(results_df)
    
    # Save to CSV
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/imputation_evaluation.csv', index=False)
    print("\nResults saved to results/imputation_evaluation.csv")

if __name__ == "__main__":
    main()
