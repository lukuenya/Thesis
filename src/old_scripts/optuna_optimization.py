import data_loader
import config

import numpy as np
from functools import partial
import optuna
import joblib
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
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
            #'scale_pos_weight': class_weight_ratio,

            'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 30, 300),
            'num_leaves': trial.suggest_int('num_leaves', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'gbdt'])
        }

    elif model_name == "xgboost":
        return {
            'booster': 'dart',
            'objective': 'binary:logistic',
            # 'scale_pos_weight': 3.585365,  # Using Fried ratio as it's more imbalanced
            'max_delta_step': 1,  # Helps with class imbalance
            'eval_metric': 'auc',
            'nthreads': -1,
            'random_state': 42,
            'verbosity': 0,

            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True), 
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 120),
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
            'depth': trial.suggest_int('depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True)
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


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
        
        # Calculate ROC AUC and F1 for this fold
        fold_auc = roc_auc_score(y_valid, preds)
        threshold = 0.3
        f1 = f1_score(y_valid, (preds > threshold).astype(int))
        
        # Store the F1 score for this fold
        f1_scores.append(f1)
        
        # Store the ROC AUC for this fold
        auc.append(fold_auc)

    # Calculate mean F1 score across all folds
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    # Store the mean F1 score and standard deviation for this trial
    trial.set_user_attr('mean_f1', mean_f1)
    trial.set_user_attr('std_f1', std_f1)
    trial.set_user_attr('fold_f1_scores', f1_scores)

    # Calculate mean ROC AUC across all folds
    mean_auc = np.mean(auc)
    std_auc = np.std(auc)

    # Store the mean ROC AUC and standard deviation for this trial
    trial.set_user_attr('std_auc', std_auc)
    trial.set_user_attr('fold_auc_scores', auc)
    
    # Return mean score across all folds
    return mean_auc


def optimize_model(score_type, n_trials, model_name):
    """Main function to optimize model hyperparameters"""
    # Load data
    X, y, feature_names = data_loader.load_data(score_type)

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

    # Print the best parameters
    print(f"Best parameters: {study.best_params}")

    # Print the best score
    print(f"Best AUC score: {study.best_value:.2f}")

    # Get the metrics from the best trial
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
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    final_model.fit(X, y)
    
    # Save feature importances
    if hasattr(final_model, 'feature_importances_'):
        feature_importances = final_model.feature_importances_
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        importance_filename = f"{model_name}_{score_type.lower()}_feature_importances.xlsx"
        feature_importances_df.to_excel(
            os.path.join(config.MODEL_OUTPUT, importance_filename),
            index=False
        )
        print(f"Feature importances saved to {importance_filename}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--score_type",
        type=str,
        required=True,
        choices=["FRIED", "FRAGIRE18"],
        help="Type of frailty score to predict"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=15,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lightgbm", "xgboost", "catboost"],
        help="Model to optimize"
    )
    
    args = parser.parse_args()

    optimize_model(score_type=args.score_type, n_trials=args.n_trials, model_name=args.model)

    print("\nOptimization complete!")