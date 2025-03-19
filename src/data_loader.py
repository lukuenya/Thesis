# src/data_loader.py

import pandas as pd
import numpy as np
import config
import preprocessing
import os

def load_data(target_score='FRIED', selected_features=False, imputation=True, feature_selection_method=None, n_features=10, threshold_percentile=20):
    """
    Prepare the dataset for training by loading data, preprocessing features, and selecting the target.
    
    Parameters:
    -----------
    target_score : str
        The target score to predict ('FRIED' or 'FRAGIRE18')
    selected_features : bool
        Whether to use pre-selected features
    imputation : bool
        Whether to use imputation or raw data
    feature_selection_method : str or None
        Feature selection method to use ('embedded', 'wrapper', or None)
    n_features : int
        Number of features to select for wrapper method
    threshold_percentile : int
        Percentile threshold for embedded feature selection
        
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    """
    print(f"Loading data for {target_score} with imputation={imputation}")
    
    # Load dataset
    df = pd.read_excel(config.TRAINING_FILE)
    
    # Get target for classification
    target_name = config.TARGET_COLUMNS_DICT[target_score]['classification']
    
    # Preprocess features
    if imputation:
        X = preprocessing.process_features(df)
    else:
        print("Using raw data without imputation")
        # Preprocess without imputation
        X = preprocessing.process_features(df, impute=False)
    
    # Get target
    y = df[target_name]
    
    # Feature selection
    if selected_features:
        # Get output paths based on imputation and feature selection state
        paths = config.get_output_paths(
            imputation=imputation, 
            feature_selection=feature_selection_method if feature_selection_method else "embedded"
        )
        output_dir = paths['feature_importances']
        
        # Construct prefix for filename
        prefix = ""
        if feature_selection_method == "wrapper":
            prefix = "wrapper_"
            
        # Load selected features from file
        filepath = os.path.join(
            output_dir,
            f"{prefix}selected_features_{target_score.lower()}_classification.txt"
        )
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                selected_feature_names = [line.strip() for line in f]
            print(f"Loaded {len(selected_feature_names)} pre-selected features from {filepath}")
            X = X[selected_feature_names]
        else:
            print(f"Warning: No pre-selected features found at {filepath}")
    
    # Apply feature selection on-the-fly
    elif feature_selection_method == 'embedded':
        # Load feature importances for all models
        importances = feature_selection.load_feature_importances(target_score, imputation)
        
        if not importances:
            print("Warning: No feature importance files found. Using all features.")
        else:
            # Aggregate importance scores across models
            agg_importance = feature_selection.aggregate_feature_importance(importances)
            
            # Select top features
            selected_feature_names = feature_selection.select_top_features(agg_importance, threshold_percentile)
            print(f"Selected {len(selected_feature_names)} features using embedded method")
            
            # Filter features
            X = X[selected_feature_names]
            
    elif feature_selection_method == 'wrapper':
        # Perform wrapper-based feature selection
        selected_feature_names, _ = feature_selection.wrapper_feature_selection(
            X, y, n_features=n_features
        )
        X = X[selected_feature_names]
    
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
