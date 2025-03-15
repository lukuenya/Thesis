# src/data_loader.py

import pandas as pd
import numpy as np
import config
import preprocessing
import os

def load_data(target_score='FRIED', selected_features=False):
    """
    Load the dataset and prepare X, y for model training
    
    Parameters:
    -----------
    target_score : str, optional (default='FRIED')
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    selected_features : bool, optional (default=False)
        Whether to use only selected important features
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    """
    df = pd.read_excel(config.TRAINING_FILE)
    
    # Prepare X and y based on target score
    if target_score.upper() == 'FRIED':
        X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Fried_State'], axis=1)
        y = df.Fried_State
    elif target_score.upper() == 'FRAGIRE18':
        X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Frailty_State_GFST'], axis=1)
        y = df.Frailty_State_GFST
    else:
        raise ValueError(f"Unknown target score: {target_score}")
    
    # Drop NaN rows in y and corresponding rows in X
    X = X[y.notna()]
    y = y.dropna()
    
    # # Drop folow-up columns
    # follow_up_cols = [col for col in X.columns if col.endswith('follow-up')]
    # X = X.drop(follow_up_cols, axis=1)
    
    # Drop columns with more than 60% missing values
    X = preprocessing.drop_high_missing(X, threshold=0.6)

    # Drop columns with near-zero variance
    X = preprocessing.drop_near_zero_variance(X)

    # Impute missing values
    X_imputed, _ = preprocessing.process_imputation(X.values)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

    # # Process features - drop highly correlated and non-useful features
    # X, _ = preprocessing.process_features(X)
    
    # Apply feature selection if requested
    if selected_features:
        feature_list = load_selected_features(target_score)
        if feature_list:
            X = X[feature_list]
    
    return X, y


def load_selected_features(score_type):
    """
    Load the list of selected features from file
    
    Parameters:
    -----------
    score_type : str
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    
    Returns:
    --------
    feature_list : list
        List of selected feature names
    """
    file_path = os.path.join(
        config.FEATURE_IMPORTANCE_DIR, 
        f"selected_features_{score_type.lower()}_classification.txt"
    )
    
    if not os.path.exists(file_path):
        print(f"Selected features file not found: {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        feature_list = [line.strip() for line in f if line.strip()]
    
    return feature_list

def load_raw_data():
    """
    Load the raw dataset without preprocessing
    
    Returns:
    --------
    df : pd.DataFrame
        Raw data
    """
    df = pd.read_excel(config.TRAINING_FILE)
    return df


def preprocess_data(df, score_type='FRIED', task='classification'):
    """
    Preprocess data for modeling
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw data
    score_type : str
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    task : str
        Task type: 'classification'
        
    Returns:
    --------
    X : pd.DataFrame
        Processed feature matrix
    y : pd.Series
        Target variable
    """
    target_col = config.TARGET_COLUMNS_DICT[score_type][task]
    
    # Drop necessary columns
    X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + [target_col], axis=1)
    y = df[target_col]
    
    # Drop follow-up columns
    follow_up_cols = [col for col in X.columns if col.endswith('follow-up')]
    X = X.drop(follow_up_cols, axis=1)
    
    # Drop columns with more than 60% missing values
    X = preprocessing.drop_high_missing(X, threshold=0.6)
    
    # Drop columns with near-zero variance
    X = preprocessing.drop_near_zero_variance(X)
    
    # Impute missing values
    X_imputed, _ = preprocessing.process_imputation(X.values)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    # Process features
    X, _ = preprocessing.process_features(X)
    
    return X, y
