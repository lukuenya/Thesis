import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
from lightgbm import LGBMRegressor
import data_loader
import joblib
import config
import re
import os


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


def get_important_features(X, y, params):
    """Train a model and return the most important features"""
    # Remove n_estimators from params if present
    train_params = params.copy()
    n_estimators = train_params.pop('n_estimators', 100)
    
    # Create a small validation set for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model with early stopping
    model = lgb.train(
        train_params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importance(importance_type='gain')
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # Select top features (those that account for 95% of total importance)
    total_importance = importance['importance'].sum()
    cumulative_importance = importance['importance'].cumsum() / total_importance
    important_features = importance[cumulative_importance <= 0.95]['feature'].tolist()
    
    return important_features


def objective(trial, X, y, folds, score_type):
    """Objective function for Optuna optimization"""
    # Define the hyperparameters to optimize
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 40),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 8),
        'verbose': -1
    }
    
    # Get number of estimators separately
    n_estimators = trial.suggest_int('n_estimators', 300, 800)
    
    # Initialize arrays to store scores
    scores = []
    
    # Perform k-fold cross validation
    for fold in range(5):
        # Get train and validation indices
        train_idx = folds != fold
        valid_idx = folds == fold
        
        # Create datasets using aligned indices
        train_data = lgb.Dataset(
            X.loc[train_idx], 
            label=y.loc[train_idx]
        )
        valid_data = lgb.Dataset(
            X.loc[valid_idx], 
            label=y.loc[valid_idx],
            reference=train_data
        )
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50)],
        )
        
        # Get best score
        scores.append(model.best_score['valid_0']['rmse'])
    
    # Return mean RMSE
    return np.mean(scores)


def optimize_model(score_type, n_trials=100):
    """
    Optimize LightGBM model using Optuna
    
    Parameters:
    -----------
    score_type : str
        'FRIED' or 'FRAGIRE18'
    n_trials : int
        Number of optimization trials
    
    Returns:
    --------
    dict
        Best parameters and study object
    """
    # Load the full dataset with folds
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    initial_rows = len(df)
    
    # Clean all column names
    df = clean_column_names(df)
    
    # Update config column names to match cleaned names
    cleaned_fried_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRIED_SCORE]
    cleaned_fragire_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRAILTY_SCORE]
    
    # Select appropriate fold column and target column based on score type
    fold_col = f'kfold_{score_type.lower()}'
    if score_type == 'FRIED':
        target_col = 'Fried_Score_FRIED_TOTAL_Version_1'
        cols_to_drop = cleaned_fried_cols + ['Frailty_Score_FRAGIRE18_SQ001']
    else:  # FRAGIRE18
        target_col = 'Frailty_Score_FRAGIRE18_SQ001'
        cols_to_drop = cleaned_fragire_cols + ['Fried_Score_FRIED_TOTAL_Version_1']
    
    # Drop rows where target variable is NaN
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    final_rows = len(df)
    print(f"\nDataset size for {score_type}:")
    print(f"Initial rows: {initial_rows}")
    print(f"Rows after dropping NaN targets: {final_rows}")
    print(f"Dropped rows: {initial_rows - final_rows}")

    # Drop unnecessary columns including both fold columns
    cols_to_drop_all = cols_to_drop + ['kfold_fried', 'kfold_fragire18']
    features = df.drop(cols_to_drop_all + [target_col], axis=1)
    
    # Remove any remaining non-numeric columns
    X = features.select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Keep track of fold column
    folds = df[fold_col]
    
    # Create initial parameters for feature selection
    init_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'learning_rate': 0.05,
        'max_depth': 6,
        'early_stopping_round': 50,
        'verbose': -1
    }
    
    # Get important features
    important_features = get_important_features(X, y, init_params)
    X = X[important_features]
    print(f"\nSelected {len(important_features)} important features out of {features.shape[1]} total features")
    print("\nTop 10 important features:")
    print("\n".join(important_features[:10]))

    # Create study object
    study = optuna.create_study(
        direction="minimize",
    )
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, X, y, folds, score_type),
        n_trials=n_trials
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbose': -1
    })
    
    # Train final model with best parameters
    final_model = LGBMRegressor(**best_params)
    final_model.fit(X, y)
    
    # Save the model
    model_filename = f"lightgbm_{score_type.lower()}_optimized.joblib"
    joblib.dump(final_model, os.path.join(config.MODEL_OUTPUT, model_filename))
    
    # Save study
    study_filename = f"{score_type.lower()}_study.joblib"
    joblib.dump(study, os.path.join(config.MODEL_OUTPUT, study_filename))
    
    return {
        'best_params': best_params,
        'study': study
    }


if __name__ == "__main__":
    # Run optimization for both scoring systems
    fried_results = optimize_model('FRIED', n_trials=100)
    fragire_results = optimize_model('FRAGIRE18', n_trials=100)
    
    # Save studies for later analysis
    joblib.dump(
        fried_results['study'],
        os.path.join(config.MODEL_OUTPUT, "fried_study.joblib")
    )
    joblib.dump(
        fragire_results['study'],
        os.path.join(config.MODEL_OUTPUT, "fragire18_study.joblib")
    )
