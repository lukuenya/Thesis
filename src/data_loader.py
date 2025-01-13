# src/data_loader.py

import pandas as pd
import numpy as np
import config
import preprocessing


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
