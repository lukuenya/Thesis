import os
import numpy as np
import pandas as pd
import joblib
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_curve, f1_score,
    mean_squared_error, r2_score, mean_absolute_error
)

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import config
import data_loader


# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plot_roc_curves(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot ROC curve and save to file."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {score_type} ({model_name})')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    suffix = '_selected' if selected_features else ''
    base_filename = f'roc_curve_{score_type.lower()}_{model_name}{suffix}'
    
    # Save as PNG
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f'{base_filename}.png'
    ), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot Precision-Recall curve and save to file."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {score_type} ({model_name})')
    plt.legend(loc="lower left")
    plt.grid(True)
    
    suffix = '_selected' if selected_features else ''
    base_filename = f'pr_curve_{score_type.lower()}_{model_name}{suffix}'
    
    # Save as PNG
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f'{base_filename}.png'
    ), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, score_type, model_name, selected_features=False):
    """Plot confusion matrix and save to file."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {score_type} ({model_name})')
    
    suffix = '_selected' if selected_features else ''
    base_filename = f'confusion_matrix_{score_type.lower()}_{model_name}{suffix}'
    
    # Save as PNG
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f'{base_filename}.png'
    ), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_threshold_impact(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot how F1 changes across thresholds."""
    thresholds = np.linspace(0.1, 0.9, 50)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='o')
    plt.xlabel('Classification Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'Impact of Classification Threshold - {score_type} ({model_name})')
    plt.grid(True)
    
    # Mark optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    plt.plot(optimal_threshold, optimal_f1, 'r*', markersize=15,
             label=f'Threshold = {optimal_threshold:.2f}, F1 = {optimal_f1:.2f}')
    plt.legend()
    
    suffix = '_selected' if selected_features else ''
    base_filename = f'threshold_impact_{score_type.lower()}_{model_name}{suffix}'
    
    # Save as PNG
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f'{base_filename}.png'
    ), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return optimal_threshold, optimal_f1

def plot_regression_metrics(y_true, y_pred, score_type, model_name, selected_features=False):
    """Plot regression metrics and save to file."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Create scatter plot of predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted Values - {score_type} ({model_name})\nRMSE: {rmse:.3f}')
    plt.grid(True)
    
    suffix = '_selected' if selected_features else ''
    base_filename = f'regression_metrics_{score_type.lower()}_{model_name}{suffix}'
    
    # Save as PNG
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Regression,
        f'{base_filename}.png'
    ), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'rmse': rmse}

def plot_feature_importance(model, feature_names, score_type, model_name, task, selected_features=False):
    """Plot feature importance and save to file."""
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model doesn't have feature importances")
        return
    
    # Sort features by importance and get top 10
    indices = np.argsort(importances)[::-1][:10]  # Get only top 10
    top_importances = importances[indices]
    top_features = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), top_importances)
    plt.xticks(range(len(indices)), top_features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top 10 Most Important Features - {score_type} ({model_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    base_filename = f'feature_importance_{score_type.lower()}_{model_name}{suffix}'
    
    # Save as PNG
    output_dir = (config.VISUALIZATION_OUTPUT_Classification 
                 if task == 'classification' 
                 else config.VISUALIZATION_OUTPUT_Regression)
    plt.savefig(os.path.join(output_dir, f'{base_filename}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR OPTIMIZATION
# -----------------------------------------------------------------------------

def get_hyperparameter_space(trial, model_name, task):
    """Define hyperparameter search space for each model."""
    if task == 'classification':
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
                'num_leaves': trial.suggest_int('num_leaves', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
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
                'scale_pos_weight': 4.59375,  # Using Fried ratio as it's more imbalanced

                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 0, 80),
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
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True)
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        if model_name == "lightgbm":
            return {
                'objective': 'regression' if task == 'regression' else 'binary',
                'metric': 'rmse' if task == 'regression' else 'auc',
                'verbosity': -1,
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                
                # Core Parameters
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                
                # Learning Parameters
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                
                # Regularization
                'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                
                # Sampling
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
        elif model_name == "xgboost":
            return {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'nthreads': -1,
                'random_state': 42,
                'verbosity': 0,

                'booster': trial.suggest_categorical('booster', ['dart', 'gbtree']),
                'n_estimators': trial.suggest_int('n_estimators', 150, 300),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 100, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
        elif model_name == "catboost":
            return {
                'eval_metric': 'RMSE',
                'random_seed': 42,
                'thread_count': -1,
                'verbose': False,
                'loss_function': 'RMSE',

                'iterations': trial.suggest_int('iterations', 30, 300),
                'depth': trial.suggest_int('depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True)
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")


def initialize_model(model_name, task, params):
    """Initialize model based on task and model name."""
    if task == 'classification':
        if model_name == "lightgbm":
            return LGBMClassifier(**params)
        elif model_name == "xgboost":
            return XGBClassifier(**params)
        elif model_name == "catboost":
            return CatBoostClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        if model_name == "lightgbm":
            return LGBMRegressor(**params)
        elif model_name == "xgboost":
            return XGBRegressor(**params)
        elif model_name == "catboost":
            return CatBoostRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")


def objective(trial, X, y, model_name, task):
    """Optuna objective function for hyperparameter optimization via CV."""
    params = get_hyperparameter_space(trial, model_name, task)
    
    if task == 'classification':
        # For classification, round the target values and convert to int
        y_int = np.round(y).astype(int)
        
        # Get class distribution for fold calculation
        _, counts = np.unique(y_int, return_counts=True)
        
        if model_name == 'lightgbm':
            model = LGBMClassifier(**params)
        elif model_name == 'xgboost':
            model = XGBClassifier(**params)
        else:
            model = CatBoostClassifier(**params)
        
        # Count samples in minority class
        min_samples = min(np.bincount(y_int))
        
        if min_samples < 3:
            # If we have less than 3 samples in any class, skip SMOTE
            pipeline = model
        else:
            # Use k_neighbors=2 (minimum possible) for small datasets
            pipeline = ImbPipeline([
                ('smote', SMOTE(k_neighbors=2, random_state=42)),
                ('clf', model)
            ])
            
        # Ensure we have at least 2 samples per class in each fold
        n_splits = min(5, min(counts))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        if model_name == 'lightgbm':
            model = LGBMRegressor(**params)
        elif model_name == 'xgboost':
            model = XGBRegressor(**params)
        else:
            model = CatBoostRegressor(**params)
        pipeline = model
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    skipped_folds = 0
    
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_int if task == 'classification' else y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold = y_int[train_idx] if task == 'classification' else y[train_idx]
        y_val_fold = y_int[val_idx] if task == 'classification' else y[val_idx]
        
        if task == 'classification':
            # Check if validation fold has at least one sample of each class
            if len(np.unique(y_val_fold)) == 1:
                print(f"Skipping fold {fold_idx + 1} due to single-class validation set")
                skipped_folds += 1
                continue
        
        try:
            pipeline.fit(X_train_fold, y_train_fold)
            
            if task == 'classification':
                y_val_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
                fold_score = roc_auc_score(y_val_fold, y_val_pred_proba)
            else:
                y_val_pred = pipeline.predict(X_val_fold)
                fold_score = -np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
            
            cv_scores.append(fold_score)
            print(f"Fold {fold_idx + 1} score: {fold_score:.4f}")
            
        except Exception as e:
            print(f"Error in fold {fold_idx + 1}: {str(e)}")
            skipped_folds += 1
            continue
    
    if len(cv_scores) == 0:
        print(f"Warning: All {n_splits} folds were skipped. Check data distribution.")
        # Return a very poor score to guide optimization away from these parameters
        return float('-inf') if task == 'classification' else float('inf')
    
    return np.mean(cv_scores)


def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold that maximizes F1 score. Only used for classification tasks."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Compute F1 for each threshold
    f1_scores = np.zeros_like(precision)
    valid_mask = (precision + recall) > 0
    f1_scores[valid_mask] = (
        2 * precision[valid_mask] * recall[valid_mask] / (precision[valid_mask] + recall[valid_mask])
    )
    
    if len(thresholds) < len(precision):
        thresholds = np.append(thresholds, 1.0)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


# -----------------------------------------------------------------------------
# MAIN TRAINING + OPTIMIZATION + PLOTTING
# -----------------------------------------------------------------------------

def optimize_model(score_type, n_trials, model_name, task='classification', selected_features=False):
    """
    1) Loads data (with or without selected features)
    2) Splits into train/test
    3) Runs Optuna CV on train only
    4) Retrains on entire train set with best params
    5) Evaluates on hold-out test set
    6) Saves & plots everything
    7) Saves final model + feature importances
    """
    print(f"\nOptimizing {model_name} for {score_type} ({task})...")
    
    # Load data
    if selected_features:
        X, y, feature_names = data_loader.load_data_with_selected_features(score_type, task)
    else:
        X, y, feature_names = data_loader.load_data(score_type, task)
    
    # Train/test split
    if task == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    
    # Create output directories if they don't exist
    if task == 'classification':
        os.makedirs(config.MODEL_OUTPUT_classification, exist_ok=True)
        os.makedirs(config.VISUALIZATION_OUTPUT_Classification, exist_ok=True)
    else:
        os.makedirs(config.MODEL_OUTPUT_regression, exist_ok=True)
        os.makedirs(config.VISUALIZATION_OUTPUT_Regression, exist_ok=True)
    
    # Run hyperparameter optimization
    study = optuna.create_study(direction='maximize' if task == 'classification' else 'minimize')
    objective_func = partial(objective, X=X_train, y=y_train, model_name=model_name, task=task)
    study.optimize(objective_func, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")
    
    # Final model training with integrated oversampling pipeline for classification
    if task == 'classification':
        # For classification, round the target values and convert to int
        y_train_int = np.round(y_train).astype(int)
        if model_name == 'lightgbm':
            final_model = LGBMClassifier(**best_params)
        elif model_name == 'xgboost':
            final_model = XGBClassifier(**best_params)
        else:
            final_model = CatBoostClassifier(**best_params)
        
        # Count samples in minority class
        min_samples = min(np.bincount(y_train_int))
        
        if min_samples < 3:
            # If we have less than 3 samples in any class, skip SMOTE
            pipeline_final = final_model
        else:
            # Use k_neighbors=2 (minimum possible) for small datasets
            pipeline_final = ImbPipeline([
                ('smote', SMOTE(k_neighbors=2, random_state=42)),
                ('clf', final_model)
            ])
        
        pipeline_final.fit(X_train, y_train_int)
        
        model_output = config.MODEL_OUTPUT_classification
        suffix = '_selected' if selected_features else ''
        model_path = os.path.join(
            model_output,
            f"{model_name}_{score_type.lower()}{suffix}.joblib"
        )
        joblib.dump(pipeline_final, model_path)
        print(f"Model pipeline saved to {model_path}")
        
        y_pred_proba = pipeline_final.predict_proba(X_test)[:, 1]
        optimal_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        plot_roc_curves(y_test, y_pred_proba, score_type, model_name, selected_features)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:
        if model_name == 'lightgbm':
            final_model = LGBMRegressor(**best_params)
        elif model_name == 'xgboost':
            final_model = XGBRegressor(**best_params)
        else:
            final_model = CatBoostRegressor(**best_params)
        
        final_model.fit(X_train, y_train)
        model_output = config.MODEL_OUTPUT_regression
        suffix = '_selected' if selected_features else ''
        model_path = os.path.join(
            model_output,
            f"{model_name}_{score_type.lower()}{suffix}.joblib"
        )
        joblib.dump(final_model, model_path)
        print(f"Model saved to {model_path}")
        
        y_pred = final_model.predict(X_test)
        metrics = plot_regression_metrics(y_test, y_pred, score_type, model_name, selected_features)
        print("\nRegression Metrics:")
        print(f"RMSE: {metrics['rmse']:.3f}")

    # Plot feature importance using the classifier from the pipeline (for classification)
    if task == 'classification':
        clf_for_importance = pipeline_final.named_steps['clf']
    else:
        clf_for_importance = final_model
    plot_feature_importance(clf_for_importance, feature_names, score_type, model_name, task, selected_features)

    return final_model if task == 'regression' else pipeline_final


# -----------------------------------------------------------------------------
# CLI ENTRY
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and optimize models')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18', 'CHUTE_6M', 'CHUTE_12M'],
                      help='Score type to use for model training')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost'],
                      help='Model to use')
    parser.add_argument('--task', type=str, default='classification',
                      choices=['classification', 'regression'],
                      help='Task type (classification or regression)')
    parser.add_argument('--n_trials', type=int, default=25,
                      help='Number of trials for hyperparameter optimization')
    parser.add_argument('--selected_features', action='store_true',
                      help='Use only selected important features')
    args = parser.parse_args()
    
    optimize_model(
        args.score_type,
        args.n_trials,
        args.model_name,
        args.task,
        args.selected_features
    )
