# src/train.py

# Import libraries
import os
import pandas as pd
import numpy as np
from sklearn import metrics
import joblib
import config
import model_dispatcher
import lightgbm as lgb
import re


def clean_name(name):
    """Clean a single column name to be compatible with LightGBM"""
    # Replace special characters and spaces with underscore
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', str(name))
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Replace multiple underscores with single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned


def clean_column_names(df):
    """Clean column names to be compatible with LightGBM"""
    df.columns = [clean_name(col) for col in df.columns]
    return df


def train_model(X_train, y_train, X_val=None, y_val=None, params=None):
    """Train a LightGBM model with the given parameters"""
    train_data = lgb.Dataset(X_train, label=y_train)
    
    if X_val is not None and y_val is not None:
        valid_data = lgb.Dataset(X_val, label=y_val)
        valid_sets = [train_data, valid_data]
        valid_names = ['train', 'valid']
    else:
        valid_sets = [train_data]
        valid_names = ['train']
    
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'num_leaves': 31,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'max_depth': 6,
            'early_stopping_round': 50,
            'verbose': -1
        }
    
    # Create callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=params.get('early_stopping_round', 50), 
                         verbose=False),
        lgb.log_evaluation(period=100)  # Reduce logging frequency
    ]
    
    # Train the model
    model = lgb.train(
        params,
        train_data,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
        num_boost_round=params.get('n_estimators', 500)
    )
    
    return model


def train(fold, score_type='FRIED'):
    """
    Train a model on the specified fold
    
    Parameters:
    -----------
    fold : int
        The fold number to use for validation
    score_type : str
        Either 'FRIED' or 'FRAGIRE18'
    """
    # Load the training data with folds
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    
    # Clean all column names
    #df = clean_column_names(df)
    
    # Update config column names to match cleaned names
    cleaned_fried_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRIED_SCORE]
    cleaned_fragire_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRAILTY_SCORE]
    
    # Clean target column names
    fried_target = clean_name('Fried_Score_FRIED_TOTAL_Version_1')
    fragire_target = clean_name('Frailty_Score_FRAGIRE18_SQ001')
    
    # Drop rows where target variable is NaN
    if score_type == 'FRIED':
        df = df.dropna(subset=[fried_target])
    else:  # FRAGIRE18
        df = df.dropna(subset=[fragire_target])
    
    # Split into training and validation
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    if score_type == 'FRIED':
        # Prepare features and target for FRIED score
        train_features = df_train.drop(cleaned_fried_cols + [fragire_target, 'kfold'], axis=1)
        valid_features = df_valid.drop(cleaned_fried_cols + [fragire_target, 'kfold'], axis=1)
        y_train = df_train[fried_target]
        y_valid = df_valid[fried_target]
    else:  # FRAGIRE18
        # Prepare features and target for FRAGIRE18 score
        train_features = df_train.drop(cleaned_fragire_cols + [fried_target, 'kfold'], axis=1)
        valid_features = df_valid.drop(cleaned_fragire_cols + [fried_target, 'kfold'], axis=1)
        y_train = df_train[fragire_target]
        y_valid = df_valid[fragire_target]
    
    # Remove any remaining non-numeric columns
    x_train = train_features.select_dtypes(include=[np.number])
    x_valid = valid_features.select_dtypes(include=[np.number])
    
    # Train the model
    model = train_model(x_train, y_train, x_valid, y_valid)
    
    # Make predictions
    preds = model.predict(x_valid)
    
    # Calculate metrics
    rmse = np.sqrt(metrics.mean_squared_error(y_valid, preds))
    mae = metrics.mean_absolute_error(y_valid, preds)
    r2 = metrics.r2_score(y_valid, preds)
    
    print(f"\nFold = {fold}, Score Type = {score_type}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"R² = {r2:.4f}")
    
    return rmse, mae, r2


if __name__ == "__main__":
    # Lists to store metrics
    fried_metrics = []
    fragire_metrics = []
    
    # Train and evaluate on all folds for both scoring systems
    for fold_ in range(5):
        # FRIED score
        metrics_fried = train(fold_, 'FRIED')
        fried_metrics.append(metrics_fried)
        
        # FRAGIRE18 score
        metrics_fragire = train(fold_, 'FRAGIRE18')
        fragire_metrics.append(metrics_fragire)
    
    # Calculate and print average metrics
    fried_means = np.mean(fried_metrics, axis=0)
    fragire_means = np.mean(fragire_metrics, axis=0)
    
    print("\nAverage Metrics across all folds:")
    print("\nFRIED Score:")
    print(f"Average RMSE = {fried_means[0]:.4f}")
    print(f"Average MAE = {fried_means[1]:.4f}")
    print(f"Average R² = {fried_means[2]:.4f}")
    
    print("\nFRAGIRE18 Score:")
    print(f"Average RMSE = {fragire_means[0]:.4f}")
    print(f"Average MAE = {fragire_means[1]:.4f}")
    print(f"Average R² = {fragire_means[2]:.4f}")
