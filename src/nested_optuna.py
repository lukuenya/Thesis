import optuna
import numpy as np
import pandas as pd
import joblib
import data_loader
import os
from sklearn.model_selection import StratifiedKFold
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

# Ensure a directory exists to store models
MODEL_OUTPUT = "results"
os.makedirs(MODEL_OUTPUT, exist_ok=True)

##############################################################################
# 1. Hyperparameter Spaces
##############################################################################

def get_hyperparameter_space(trial, model_name, class_weight_ratio=3.585365):
    if model_name == "lightgbm":
        return {
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'n_jobs': -1,
            'is_unbalance': True,
            'verbosity': -1,
            'boosting_type': 'dart',
            'early_stopping_round': 1,

            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),

            'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 30, 100),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        }
    elif model_name == "xgboost":
        return {
            'booster': 'dart',
            'objective': 'binary:logistic',
            'random_state': 42,
            'nthreads': -1,
            'eval_metric': 'auc',
            'max_delta_step': 1,
            'verbosity': 0,

            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'eta': trial.suggest_float('eta', 1e-4, 0.1, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 120),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
    elif model_name == "catboost":
        return {
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'loss_function': 'Logloss',

            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True)
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")

##############################################################################
# 2. Inner CV Objective for Optuna
##############################################################################

def inner_cv_objective(trial, X_train, y_train, model_name, class_weight_ratio):
    params = get_hyperparameter_space(trial, model_name, class_weight_ratio)

    kf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for tr_idx, val_idx in kf_inner.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        if model_name == "lightgbm":
            model = LGBMClassifier(**params)
        elif model_name == "xgboost":
            model = XGBClassifier(**params)
        elif model_name == "catboost":
            model = CatBoostClassifier(**params)

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        y_val_proba = model.predict_proba(X_val)[:, 1]
        auc_scores.append(roc_auc_score(y_val, y_val_proba))

    return np.mean(auc_scores)

##############################################################################
# 3. Outer Nested CV
##############################################################################

def nested_cv_with_optuna(X, y, model_name="lightgbm", score_type="FRIED", n_outer_splits=3, n_inner_trials=15):
    outer_kf = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    
    outer_aucs, outer_f1s, best_params_list, best_thresholds = [], [], [], []

    for train_idx, test_idx in outer_kf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: inner_cv_objective(trial, X_train, y_train, model_name, class_weight_ratio=3.585365),
            n_trials=n_inner_trials
        )

        best_params = study.best_params
        best_params_list.append(best_params)

        if model_name == "lightgbm":
            final_model = LGBMClassifier(**best_params)
        elif model_name == "xgboost":
            final_model = XGBClassifier(**best_params)
        elif model_name == "catboost":
            final_model = CatBoostClassifier(**best_params)

        final_model.fit(X_train, y_train)
        y_test_proba = final_model.predict_proba(X_test)[:, 1]

        thresholds = np.linspace(0.1, 0.9, 100)
        best_f1, best_threshold = max(
            ((f1_score(y_test, (y_test_proba >= t).astype(int)), t) for t in thresholds),
            key=lambda x: x[0]
        )
        best_thresholds.append(best_threshold)

        auc = roc_auc_score(y_test, y_test_proba)
        outer_aucs.append(auc)
        outer_f1s.append(best_f1)

    mean_auc, mean_f1 = np.mean(outer_aucs), np.mean(outer_f1s)

    results = {
        'mean_auc': mean_auc,
        'mean_f1': mean_f1,
        'best_params': best_params_list,
        'best_thresholds': best_thresholds
    }

    final_model.fit(X, y)
    joblib.dump(final_model, os.path.join(MODEL_OUTPUT, f"final_model_{model_name}_{score_type}.joblib"))
    
    return results 

##############################################################################
# 5. Main Script
##############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--score_type", type=str, default="FRIED", choices=["FRIED", "FRAGIRE18"])
    parser.add_argument("--model_name", type=str, default="lightgbm", choices=["lightgbm","xgboost","catboost"])
    parser.add_argument("--n_outer_splits", type=int, default=3, help="Number of outer folds.")
    parser.add_argument("--n_inner_trials", type=int, default=15, help="Number of Optuna trials for each outer fold.")
    args = parser.parse_args()

    # Load data
    X, y, _ = data_loader.load_data(args.score_type)

    # Run Nested CV
    print("\n===== Running Nested CV with Optuna =====")
    results = nested_cv_with_optuna(
        X=X,
        y=y,
        model_name=args.model_name,
        score_type=args.score_type,
        n_outer_splits=args.n_outer_splits,
        n_inner_trials=args.n_inner_trials
    )

    print("\nNested CV Results:")
    print(results)
