# src/data_loader.py

import pandas as pd
import numpy as np
import config
import preprocessing
import os

def load_data(target_score='FRIED', task='classification'):
    """
    Load the dataset and prepare X, y for model training
    
    Parameters:
    -----------
    target_score : str, optional (default='FRIED')
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    task : str, optional (default='classification')
        Type of task: 'classification' or 'regression'
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (FRIED or FRAGIRE18 score/state)
    """
    df = pd.read_excel(config.TRAINING_FILE)
    
    if target_score.upper() == 'FRIED':
        # Drop FRIED-related columns and FRAGIRE18 score
        if task == 'classification':
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Fried_State'], axis=1)
            y = df.Fried_State
        else:  # regression
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Fried_Score_FRIED_TOTAL_Version_1'], axis=1)
            y = df.Fried_Score_FRIED_TOTAL_Version_1
    else:  # FRAGIRE18
        # Drop FRAGIRE18 score and FRIED-related columns
        if task == 'classification':
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Frailty_State_GFST'], axis=1)
            y = df.Frailty_State_GFST
        else:  # regression
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Frailty_Score_FRAGIRE18_SQ001'], axis=1)
            y = df.Frailty_Score_FRAGIRE18_SQ001

    # Drop folow-up columns
    follow_up_cols = [col for col in X.columns if col.endswith('follow-up')]
    X = X.drop(follow_up_cols, axis=1)
    
    # # Remove any remaining non-numeric columns
    # X = X.select_dtypes(include=[np.number])

    # Drop columns with more than 60% missing values
    X = preprocessing.drop_high_missing(X, threshold=0.6)

    # Drop columns with near-zero variance
    X = preprocessing.drop_near_zero_variance(X)

    # # Drop columns with high correlation
    # X = preprocessing.drop_correlated_features(X, method='pearson', correlation_threshold=0.95)

    # Get feature names
    feature_names = X.columns

    # Drop NaN rows in y and corresponding rows in X
    X = X[y.notna()]
    y = y.dropna()

    # Convert to numpy arrays for optimization
    X = X.values
    y = y.values
    
    return X, y, feature_names

def load_selected_features(score_type, task='classification'):
    """Load the list of selected features from file"""
    output_dir = (config.FEATURES_IMPORTANCE_OUTPUT_classification 
                 if task == 'classification' 
                 else config.FEATURES_IMPORTANCE_OUTPUT_regression)
    feature_file = os.path.join(output_dir, f"selected_features_{score_type.lower()}.txt")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Selected features file not found: {feature_file}")
    
    with open(feature_file, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    return selected_features

def load_data_with_selected_features(target_score='FRIED', task='classification'):
    """
    Load the dataset using only selected important features
    
    Parameters:
    -----------
    target_score : str, optional (default='FRIED')
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    task : str, optional (default='classification')
        Type of task: 'classification' or 'regression'
    
    Returns:
    --------
    X : np.array
        Feature matrix with only selected features
    y : np.array
        Target variable
    feature_names : list
        Names of selected features
    """
    # Load full dataset
    X, y, _ = load_data(target_score, task)
    
    # Load selected features
    selected_features = load_selected_features(target_score, task)
    
    # Convert X back to DataFrame with original column names
    X = pd.DataFrame(X, columns=_)
    
    # Select only important features
    X = X[selected_features]
    
    # Get feature names and convert to numpy arrays
    feature_names = X.columns
    X = X.values
    
    return X, y, feature_names
