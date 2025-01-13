import os
import re
import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.metrics import mean_squared_error
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import config

def clean_name(name):
    """Clean a single column name to be compatible with models"""
    # Replace special characters and spaces with underscore
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', str(name))
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Replace multiple underscores with single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned

def clean_column_names(df):
    """Clean column names to be compatible with models"""
    df.columns = [clean_name(col) for col in df.columns]
    return df

def get_tabnet_param_space(trial):
    """Define parameter space for TabNet"""
    return {
        # Architecture parameters
        "n_d": trial.suggest_int("n_d", 8, 64),  # Width of the decision prediction layer
        "n_a": trial.suggest_int("n_a", 8, 64),  # Width of the attention embedding for each mask
        "n_steps": trial.suggest_int("n_steps", 3, 10),  # Number of steps in the architecture
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),  # Feature reusage coefficient
        
        # Attention parameters
        "n_independent": trial.suggest_int("n_independent", 1, 5),  # Number of independent GLU layers
        "n_shared": trial.suggest_int("n_shared", 1, 5),  # Number of shared GLU layers
        
        # Sparsity parameters
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),  # Sparsity regularization
        
        # Training parameters
        "optimizer_params": {
            "lr": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        }
    }

def get_sample_weights(y, score_type):
    """Calculate sample weights to handle class imbalance"""
    if score_type == 'FRIED':
        # Group score 4.0 with 3.0 as done in fold creation
        y_modified = y.copy()
        y_modified = y_modified.replace(4.0, 3.0)
        # Calculate weights inversely proportional to class frequencies
        class_weights = dict(1/np.bincount(y_modified.astype(int)))
        weights = np.array([class_weights[int(val)] for val in y_modified])
    else:  # FRAGIRE18
        # Group score 4.0 with 5.0 as done in fold creation
        y_modified = y.copy()
        y_modified = y_modified.replace(4.0, 5.0)
        # Calculate weights inversely proportional to class frequencies
        class_weights = dict(1/np.bincount(y_modified.astype(int)))
        weights = np.array([class_weights[int(val)] for val in y_modified])
    
    # Normalize weights to sum to len(y)
    weights = weights * (len(y) / weights.sum())
    return weights

def objective(trial, X, y, folds, score_type, test_fold):
    """Objective function for Optuna optimization"""
    params = get_tabnet_param_space(trial)
    scores = []
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Calculate sample weights for the entire dataset
    weights = get_sample_weights(y, score_type)
    
    # Perform k-fold cross validation (excluding test fold)
    for fold in range(5):
        if fold == test_fold:  # Skip the test fold
            continue
            
        train_idx = folds[folds != fold].index
        valid_idx = folds[folds == fold].index
        
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        w_train = weights[train_idx]
        
        # Initialize model with trial parameters
        model = TabNetRegressor(
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            n_independent=params['n_independent'],
            n_shared=params['n_shared'],
            lambda_sparse=params['lambda_sparse'],
            optimizer_params=params['optimizer_params'],
            device_name=device
        )
        
        # Train model with sample weights
        model.fit(
            X_train.values, y_train.values,
            eval_set=[(X_valid.values, y_valid.values)],
            weights=w_train,
            patience=10,
            max_epochs=100,
            eval_metric=['rmse']
        )
        
        # Make predictions
        y_pred = model.predict(X_valid.values)
        
        # Calculate weighted RMSE for validation
        valid_weights = weights[valid_idx]
        valid_weights_normalized = valid_weights / valid_weights.sum()
        fold_score = np.sqrt(np.average((y_valid - y_pred) ** 2, weights=valid_weights_normalized))
        scores.append(fold_score)
    
    return np.mean(scores)

def optimize_tabnet(score_type, n_trials=100, test_fold=4):
    """Optimize TabNet model using Optuna"""
    print(f"\nOptimizing TabNet for {score_type}...")
    
    # Load data
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    initial_rows = len(df)
    
    # Clean column names
    df = clean_column_names(df)
    
    # Update config column names to match cleaned names
    cleaned_fried_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRIED_SCORE]
    cleaned_fragire_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRAILTY_SCORE]
    
    # Get target column name based on score type
    target_col = clean_name('Fried_Score_FRIED_TOTAL_Version_1') if score_type == 'FRIED' else clean_name('Frailty_Score_FRAGIRE18_SQ001')
    fold_col = clean_name(f'kfold_{score_type.lower()}')
    
    # Drop rows with NaN in target
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    final_rows = len(df)
    
    print(f"\nDataset size for {score_type}:")
    print(f"Initial rows: {initial_rows}")
    print(f"Rows after dropping NaN targets: {final_rows}")
    print(f"Dropped rows: {initial_rows - final_rows}")
    
    # Get features and target
    cols_to_drop = cleaned_fried_cols if score_type == 'FRIED' else cleaned_fragire_cols
    X = df.drop(cols_to_drop + [target_col], axis=1)
    y = df[target_col]
    folds = df[fold_col]
    
    # Calculate class distribution
    print("\nClass distribution:")
    value_counts = y.value_counts().sort_index()
    for val, count in value_counts.items():
        print(f"Score {val}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Split into train and test using predefined folds
    train_idx = folds[folds != test_fold].index
    test_idx = folds[folds == test_fold].index
    
    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]
    folds_train = folds.loc[train_idx]
    
    print(f"\nUsing fold {test_fold} as test set:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Calculate sample weights
    train_weights = get_sample_weights(y_train, score_type)
    test_weights = get_sample_weights(y_test, score_type)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create study
    study = optuna.create_study(direction="minimize")
    
    # Optimize using only training data
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, folds_train, score_type, test_fold),
        n_trials=n_trials
    )
    
    # Get best parameters
    best_params = study.best_params
    
    # Train final model with best parameters
    final_model = TabNetRegressor(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        n_independent=best_params['n_independent'],
        n_shared=best_params['n_shared'],
        lambda_sparse=best_params['lambda_sparse'],
        optimizer_params={"lr": best_params['learning_rate']},
        device_name=device
    )
    
    # Fit final model on all training data with weights
    final_model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_train.values, y_train.values)],  # Use same data for validation to track training
        weights=train_weights,
        patience=20,  # Increase patience for final model
        max_epochs=200  # Increase epochs for final model
    )
    
    # Evaluate on test set
    test_predictions = final_model.predict(X_test.values)
    
    # Calculate both weighted and unweighted RMSE
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
    test_weights_normalized = test_weights / test_weights.sum()
    weighted_test_rmse = np.sqrt(np.average((y_test - test_predictions) ** 2, weights=test_weights_normalized))
    
    print(f"\nTest Set (Fold {test_fold}) Metrics:")
    print(f"Unweighted RMSE: {test_rmse:.4f}")
    print(f"Weighted RMSE: {weighted_test_rmse:.4f}")
    
    # Print per-class errors
    print("\nPer-class Mean Absolute Error:")
    for score in sorted(y_test.unique()):
        mask = y_test == score
        if mask.any():
            mae = np.abs(y_test[mask] - test_predictions[mask]).mean()
            count = mask.sum()
            print(f"Score {score}: MAE = {mae:.4f} (n={count})")
    
    # Save model and study
    model_filename = f"tabnet_{score_type.lower()}_optimized.joblib"
    study_filename = f"{score_type.lower()}_tabnet_study.joblib"
    
    # Save model, study, and test indices for later analysis
    save_dict = {
        "model": final_model,
        "test_fold": test_fold,
        "test_indices": test_idx,
        "test_rmse": test_rmse,
        "weighted_test_rmse": weighted_test_rmse,
        "best_params": best_params,
        "train_weights": train_weights,
        "test_weights": test_weights
    }
    
    joblib.dump(save_dict, os.path.join(config.MODEL_OUTPUT, model_filename))
    joblib.dump(study, os.path.join(config.MODEL_OUTPUT, study_filename))
    
    return {
        "best_params": best_params,
        "best_value": study.best_value,
        "study": study,
        "test_rmse": test_rmse,
        "weighted_test_rmse": weighted_test_rmse
    }

if __name__ == "__main__":
    # Use the last fold (4) as test set
    test_fold = 4
    
    try:
        print("\nOptimizing TabNet...")
        fried_results = optimize_tabnet('FRIED', n_trials=100, test_fold=test_fold)
        fragire_results = optimize_tabnet('FRAGIRE18', n_trials=100, test_fold=test_fold)
    except Exception as e:
        print(f"Error optimizing TabNet: {str(e)}")
