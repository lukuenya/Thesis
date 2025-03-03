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
        Which score to predict: 'FRIED', 'FRAGIRE18', 'CHUTE_6M', or 'CHUTE_12M'
    task : str, optional (default='classification')
        Type of task: 'classification' or 'regression'
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    feature_names : list
        Names of features
    """
    df = pd.read_excel(config.TRAINING_FILE)
    
    # Prepare X and y based on target score and task
    if target_score.upper() == 'FRIED':
        if task == 'classification':
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Fried_State'], axis=1)
            y = df.Fried_State
        else:  # regression
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Fried_Score_FRIED_TOTAL_Version_1'], axis=1)
            y = df.Fried_Score_FRIED_TOTAL_Version_1

    elif target_score.upper() == 'FRAGIRE18':
        if task == 'classification':
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Frailty_State_GFST'], axis=1)
            y = df.Frailty_State_GFST
        else:  # regression
            X = df.drop(config.COLS_TO_DROP_FRAGIRE18_FRIED + ['Frailty_Score_FRAGIRE18_SQ001'], axis=1)
            y = df.Frailty_Score_FRAGIRE18_SQ001

    elif target_score.upper() == 'CHUTE_6M':
        if task != 'classification':
            raise ValueError("Falls prediction only supports classification task")
        X = df.drop(config.COLS_TO_DROP_CHUTES + [config.TARGET_CHUTE_6M], axis=1)
        y = df[config.TARGET_CHUTE_6M]

    elif target_score.upper() == 'CHUTE_12M':
        if task != 'classification':
            raise ValueError("Falls prediction only supports classification task")
        X = df.drop(config.COLS_TO_DROP_CHUTES + [config.TARGET_CHUTE_12M], axis=1)
        y = df[config.TARGET_CHUTE_12M]
    else:
        raise ValueError(f"Unknown target score: {target_score}")
    

    # Drop NaN rows in y and corresponding rows in X
    X = X[y.notna()]
    y = y.dropna()
    
    # Drop folow-up columns
    follow_up_cols = [col for col in X.columns if col.endswith('follow-up')]
    X = X.drop(follow_up_cols, axis=1)
    
    # Drop columns with more than 60% missing values
    X = preprocessing.drop_high_missing(X, threshold=0.6)

    # Drop columns with near-zero variance
    X = preprocessing.drop_near_zero_variance(X)

    # Store feature names before imputation
    feature_names = X.columns.tolist()

    # Apply the new imputation strategy
    X_imputed, _ = preprocessing.process_imputation(X, verbose=False)
    X = pd.DataFrame(X_imputed, columns=feature_names, index=X.index)

    # # Get feature names
    # feature_names = X.columns

    # # Drop NaN rows in y and corresponding rows in X
    # X = X[y.notna()]
    # y = y.dropna()

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
        Which score to predict: 'FRIED', 'FRAGIRE18', 'CHUTE_6M', or 'CHUTE_12M'
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
    # First load all data with imputation
    X, y, all_features = load_data(target_score, task)
    
    # Then load selected features
    selected_features = load_selected_features(target_score, task)
    
    if selected_features is None or len(selected_features) == 0:
        raise ValueError(f"No selected features found for {target_score} ({task})")
    
    # Convert X back to DataFrame with original column names
    X = pd.DataFrame(X, columns=all_features)
    
    # Select only the important features
    X = X[selected_features]
    
    # Get feature names and convert to numpy arrays
    feature_names = X.columns
    X = X.values
    
    return X, y, feature_names
