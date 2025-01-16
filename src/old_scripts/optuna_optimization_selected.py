import data_loader
import config
import numpy as np
from functools import partial
import optuna
import joblib
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    precision_recall_curve
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def get_hyperparameter_space(trial, model_name):
    """
    Define hyperparameter search space for each model
    """
    if model_name == "lightgbm":
        return {
            'objective': 'binary',
            'is_unbalance': True,
            'metric': 'auc',
            'n_jobs': -1,
            'random_state': 42,
            'verbosity': -1,

            'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 30, 300),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),  # Reduced from 500
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Reduced from 12
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'gbdt'])
        }

    elif model_name == "xgboost":
        return {
            'booster': 'dart',
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthreads': -1,
            'random_state': 42,
            'verbosity': 0,

            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),  # Reduced from 10
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 80),  # Reduced from 120
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }

    elif model_name == "catboost":
        return {
            'eval_metric': 'AUC',
            'random_seed': 42,
            'thread_count': -1,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'loss_function': 'Logloss',

            'iterations': trial.suggest_int('iterations', 30, 300),
            'depth': trial.suggest_int('depth', 3, 8),  # Reduced from 12
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True)
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

def find_optimal_threshold(y_true, y_pred):
    """Find optimal threshold for F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Handle case where precision + recall = 0
    f1_scores = np.zeros_like(precision)
    valid_mask = (precision + recall) > 0
    f1_scores[valid_mask] = 2 * precision[valid_mask] * recall[valid_mask] / (precision[valid_mask] + recall[valid_mask])
    
    # Handle case where thresholds is shorter than precision by 1
    if len(thresholds) < len(precision):
        thresholds = np.append(thresholds, 1.0)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # If we still get NaN, fallback to default threshold
    if np.isnan(optimal_threshold) or np.isnan(optimal_f1):
        y_pred_binary = (y_pred > 0.3).astype(int)
        optimal_threshold = 0.3
        optimal_f1 = f1_score(y_true, y_pred_binary)
    
    return optimal_threshold, optimal_f1

def objective(trial, x, y, model_name):
    """Optuna objective function for hyperparameter optimization"""
    # Get hyperparameters for this trial
    params = get_hyperparameter_space(trial, model_name)
    
    # Perform k-fold cross validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc = []
    f1_scores = []

    for idx in kf.split(X=x, y=y):
        train_idx, valid_idx = idx[0], idx[1]
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]
        
        # Create a new model instance for each fold
        if model_name == "lightgbm":
            model = LGBMClassifier(**params)
        elif model_name == "xgboost":
            model = XGBClassifier(**params)
        elif model_name == "catboost":
            model = CatBoostClassifier(**params)
            
        # Train the model
        model.fit(x_train, y_train)
        
        # Predict probabilities
        preds = model.predict_proba(x_valid)[:, 1]
        
        # Calculate ROC AUC for this fold
        fold_auc = roc_auc_score(y_valid, preds)
        
        # Find optimal threshold for this fold
        optimal_threshold, fold_f1 = find_optimal_threshold(y_valid, preds)
        
        # Store the scores
        f1_scores.append(fold_f1)
        auc.append(fold_auc)

    # Calculate and store metrics
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    mean_auc = np.mean(auc)
    std_auc = np.std(auc)
    
    # Store all metrics as trial attributes
    trial.set_user_attr('mean_f1', mean_f1)
    trial.set_user_attr('std_f1', std_f1)
    trial.set_user_attr('fold_f1_scores', f1_scores)
    trial.set_user_attr('std_auc', std_auc)
    trial.set_user_attr('fold_auc_scores', auc)
    
    return mean_auc

def optimize_model(score_type, n_trials, model_name):
    """Main function to optimize model hyperparameters"""
    # Load data with selected features
    X, y, feature_names = data_loader.load_data_with_selected_features(score_type)
    print(f"Training with {len(feature_names)} selected features")

    # Create partial function with fixed parameters
    optimization_function = partial(
        objective,
        x=X,
        y=y,
        model_name=model_name
    )
    
    # Create and run study
    study = optuna.create_study(direction='maximize')
    study.optimize(optimization_function, n_trials=n_trials)

    # Print results
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best AUC score: {study.best_value:.2f}")

    # Get metrics from best trial
    best_trial = study.best_trial
    std_auc = best_trial.user_attrs['std_auc']
    mean_f1 = best_trial.user_attrs['mean_f1']
    std_f1 = best_trial.user_attrs['std_f1']
    
    print(f"Mean F1 score: {mean_f1:.2f}")
    print(f"Standard deviation of F1: {std_f1:.2f}")
    print(f"Standard deviation of AUC: {std_auc:.2f}")

    # Train final model with best parameters
    best_params = study.best_params
    if model_name == "lightgbm":
        final_model = LGBMClassifier(**best_params)
    elif model_name == "xgboost":
        final_model = XGBClassifier(**best_params)
    elif model_name == "catboost":
        final_model = CatBoostClassifier(**best_params)
    
    # Perform final cross-validation evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_thresholds = []
    cv_f1_scores = []
    cv_auc_scores = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        final_model.fit(X_train, y_train)
        
        # Get predictions
        val_proba = final_model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold and scores for this fold
        fold_threshold, fold_f1 = find_optimal_threshold(y_val, val_proba)
        fold_auc = roc_auc_score(y_val, val_proba)
        
        cv_thresholds.append(fold_threshold)
        cv_f1_scores.append(fold_f1)
        cv_auc_scores.append(fold_auc)
    
    # Calculate final metrics
    final_threshold = np.mean(cv_thresholds)
    final_f1 = np.mean(cv_f1_scores)
    final_auc = np.mean(cv_auc_scores)
    
    print(f"\nFinal Cross-Validation Metrics:")
    print(f"Mean AUC: {final_auc:.2f} (±{np.std(cv_auc_scores):.2f})")
    print(f"Mean F1: {final_f1:.2f} (±{np.std(cv_f1_scores):.2f})")
    print(f"Mean Threshold: {final_threshold:.2f} (±{np.std(cv_thresholds):.2f})")
    
    # Train final model on full dataset
    final_model.fit(X, y)
    
    # Save the model and metrics
    model_filename = f"{model_name}_{score_type.lower()}_selected_features.joblib"
    save_dict = {
        'model': final_model,
        'optimal_threshold': final_threshold,
        'feature_names': feature_names,
        'cv_metrics': {
            'auc_mean': final_auc,
            'auc_std': np.std(cv_auc_scores),
            'f1_mean': final_f1,
            'f1_std': np.std(cv_f1_scores),
            'threshold_mean': final_threshold,
            'threshold_std': np.std(cv_thresholds)
        }
    }
    joblib.dump(save_dict, os.path.join(config.MODEL_OUTPUT, model_filename))

    # Save feature importances
    feature_importances = final_model.feature_importances_
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    feature_importances_df.to_excel(
        os.path.join(config.MODEL_OUTPUT, f"{model_name}_{score_type.lower()}_selected_features_importance.xlsx"),
        index=False
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18'],
                      help='Which score to use for optimization')
    parser.add_argument('--n_trials', type=int, default=100,
                      help='Number of optimization trials')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost'],
                      help='Which model to optimize')
    
    args = parser.parse_args()
    optimize_model(args.score_type, args.n_trials, args.model_name)