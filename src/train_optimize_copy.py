import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

import data_loader
import config

def plot_roc_curve(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot ROC curve for binary classification"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(6, 4), dpi=300)  
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the figure
    output_dir = config.VISUALIZATION_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        output_dir, 
        f"roc_curve_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot precision-recall curve for binary classification"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(6, 4), dpi=300)  
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.axhline(y=np.sum(y_true) / len(y_true), color='navy', linestyle='--', label=f'Baseline (AP = {np.sum(y_true) / len(y_true):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    # Save the figure
    output_dir = config.VISUALIZATION_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        output_dir, 
        f"precision_recall_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, score_type, model_name, selected_features=False):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4), dpi=300)  
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save the figure
    output_dir = config.VISUALIZATION_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        output_dir, 
        f"confusion_matrix_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()

def plot_threshold_impact(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot impact of classification threshold on precision, recall, and F1 score"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred))
        recalls.append(recall_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    plt.figure(figsize=(6, 4), dpi=300)  
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, recalls, 'g-', label='Recall')
    plt.plot(thresholds, f1_scores, 'r-', label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_dir = config.VISUALIZATION_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        output_dir, 
        f"threshold_impact_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()

def plot_feature_importance(feature_names, importances, score_type, model_name, selected_features=False):
    """Plot feature importance for the model"""
    # Sort features by importance
    indices = np.argsort(importances)[-10:]  # Get top 10 features
    
    plt.figure(figsize=(6, 4), dpi=300)  
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    
    # Save the figure
    output_dir = config.VISUALIZATION_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    
    suffix = '_selected' if selected_features else ''
    base_filename = f"feature_importance_{model_name}_{score_type.lower()}{suffix}"
    
    plt.savefig(os.path.join(output_dir, f'{base_filename}.png'),
               bbox_inches='tight', dpi=300)
    plt.close()

def find_optimal_threshold(y_true, y_pred_proba):
    """Find the optimal threshold for binary classification"""
    thresholds = np.linspace(0.1, 0.9, 100)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR OPTIMIZATION
# -----------------------------------------------------------------------------

def get_hyperparameter_space(trial, model_name):
    """Define hyperparameter search space for different models"""
    
    if model_name == 'lightgbm':
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.001, 0.1),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'is_unbalance': True
        }
        return params
        
    elif model_name == 'xgboost':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'eta': trial.suggest_float('eta', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'random_state': 42,
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
        return params
        
    elif model_name == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
            'random_seed': 42,
            'task_type': 'CPU',
            'verbose': 0,
            'auto_class_weights': 'Balanced'
        }
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
        return params
        
    elif model_name == 'randomforest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        return params
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def initialize_model(model_name, params):
    """Initialize a model with given parameters"""
    if model_name == 'lightgbm':
        return LGBMClassifier(**params)
    elif model_name == 'xgboost':
        return XGBClassifier(**params)
    elif model_name == 'catboost':
        return CatBoostClassifier(**params)
    elif model_name == 'randomforest':
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
def objective(trial, X, y, model_name, score_type, cv_folds=5):
    """Objective function for hyperparameter optimization"""
    # Get hyperparameters from trial
    params = get_hyperparameter_space(trial, model_name)
    
    # Initialize model
    model = initialize_model(model_name, params)
    
    # Create pipeline with SMOTETomek for handling class imbalance
    pipeline = Pipeline([
        ('smote', SMOTETomek(random_state=42)),
        ('model', model)
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Use roc_auc score for optimization (higher is better)
    scores = cross_val_score(
        pipeline, X, y, 
        scoring='roc_auc', 
        cv=cv, 
        n_jobs=-1
    )
    
    # Return mean score
    return scores.mean()

def optimize_model(score_type, n_trials=15, model_name='lightgbm', selected_features=False):
    """
    Optimize model hyperparameters using Optuna
    
    Parameters:
    -----------
    score_type : str
        Which score to predict: 'FRIED', 'FRAGIRE18'
    n_trials : int, optional (default=15)
        Number of trials for hyperparameter optimization
    model_name : str, optional (default='lightgbm')
        Model to use: 'lightgbm', 'xgboost', 'catboost', or 'randomforest'
    selected_features : bool, optional (default=False)
        Whether to use only selected important features
    
    Returns:
    --------
    model : Pipeline
        Trained pipeline with the best hyperparameters
    best_params : dict
        Best hyperparameters found
    """
    # Load data
    if selected_features:
        X, y = data_loader.load_data(score_type, selected_features=True)
    else:
        X, y = data_loader.load_data(score_type)
    
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y = y.values
    
    # Create Optuna study for maximizing score
    study = optuna.create_study(direction="maximize")
    
    # Define and set custom objective
    custom_objective = lambda trial: objective(trial, X, y, model_name, score_type)
    
    # Run optimization
    print(f"Starting optimization for {model_name} model predicting {score_type}...")
    study.optimize(custom_objective, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best score: {study.best_value:.4f}")
    
    # Create output directories
    os.makedirs(config.MODEL_OUTPUT, exist_ok=True)
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    
    # Save study results
    model_info = {
        'model_name': model_name,
        'score_type': score_type,
        'best_params': best_params,
        'best_score': study.best_value,
        'feature_names': feature_names
    }
    
    # Add suffix if using selected features
    suffix = '_selected' if selected_features else ''
    
    # Save model info
    with open(os.path.join(
        config.MODEL_OUTPUT, 
        f"{model_name}_{score_type.lower()}{suffix}_info.pkl"
    ), 'wb') as f:
        pickle.dump(model_info, f)
    
    # Train final model on all data
    # Handle class imbalance with SMOTETomek
    model = initialize_model(model_name, best_params)
    pipeline_final = Pipeline([
        ('smote', SMOTETomek(random_state=42)),
        ('model', model)
    ])
    
    # Fit the model
    pipeline_final.fit(X, y)
    
    # Save feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        plot_feature_importance(feature_names, importances, score_type, model_name, selected_features)
    
    # Save full model
    joblib.dump(
        pipeline_final, 
        os.path.join(
            config.MODEL_OUTPUT, 
            f"{model_name}_{score_type.lower()}{suffix}_model.joblib"
        )
    )
    
    return pipeline_final, best_params

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and optimize models')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18'],
                      help='Score type to use for model training')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost', 'randomforest'],
                      help='Model to use')
    parser.add_argument('--n_trials', type=int, default=25,
                      help='Number of trials for hyperparameter optimization')
    parser.add_argument('--selected_features', action='store_true',
                      help='Use only selected important features')
    args = parser.parse_args()
    
    optimize_model(
        args.score_type,
        args.n_trials,
        args.model_name,
        args.selected_features
    )
