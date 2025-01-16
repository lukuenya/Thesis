# src/data_loader.py

import pandas as pd
import numpy as np
import config
import preprocessing
import os

def load_data(target_score='FRIED'):
    """
    Load the dataset and prepare X, y for model training
    
    Parameters:
    -----------
    target_score : str, optional (default='FRIED')
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (FRIED or FRAGIRE18 score)
    """
    df = pd.read_excel(config.TRAINING_FILE)
    
    if target_score.upper() == 'FRIED':
        # Drop FRIED-related columns and FRAGIRE18 score
        X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Fried_State'], axis=1)
        y = df.Fried_State
    else:  # FRAGIRE18
        # Drop FRAGIRE18 score and FRIED-related columns
        X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Frailty_State_GFST'], axis=1)
        y = df.Frailty_State_GFST

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

def load_selected_features(score_type):
    """Load the list of selected features from file"""
    feature_file = os.path.join(config.FEATURES_IMPORTANCE_OUTPUT, f"selected_features_{score_type.lower()}.txt")
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Selected features file not found: {feature_file}")
    
    with open(feature_file, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    return selected_features

def load_data_with_selected_features(target_score='FRIED'):
    """
    Load the dataset using only selected important features
    
    Parameters:
    -----------
    target_score : str, optional (default='FRIED')
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    
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
    X, y, all_features = load_data(target_score)
    
    # Load selected features
    selected_features = load_selected_features(target_score)
    
    # Convert all_features (Index) to list and get indices of selected features
    all_features_list = all_features.tolist()
    selected_indices = [all_features_list.index(feat) for feat in selected_features]
    
    # Select only the important features
    X_selected = X[:, selected_indices]
    
    return X_selected, y, selected_features
