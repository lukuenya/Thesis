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
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import config
import data_loader


# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (same style as you had)
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
        elif model_name == "randomforest":
            return {
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced',
                
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
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
        elif model_name == "randomforest":
            return {
                'random_state': 42,
                'n_jobs': -1,
                
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson'])
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
        elif model_name == "randomforest":
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    else:
        if model_name == "lightgbm":
            return LGBMRegressor(**params)
        elif model_name == "xgboost":
            return XGBRegressor(**params)
        elif model_name == "catboost":
            return CatBoostRegressor(**params)
        elif model_name == "randomforest":
            return RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")


def objective(trial, X, y, model_name, task):
    """Optuna objective function for hyperparameter optimization via CV."""
    params = get_hyperparameter_space(trial, model_name, task)
    
    # Initialize model based on task and model name
    model = initialize_model(model_name, task, params)
    
    # Prepare cross-validation
    if task == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Convert y to integers for SMOTETomek
        y_int = y.astype(int)
        # Count samples in each class
        counts = np.bincount(y_int)
        min_samples = min(counts)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Check if validation fold has only one class
        if task == 'classification' and len(np.unique(y_val_fold)) == 1:
            continue  # Skip this fold
        
        # Train model
        if task == 'classification':
            # Apply SMOTETomek for class imbalance if we have enough samples
            if min_samples >= 3:
                # Use SMOTETomek (SMOTE + Tomek links)
                smote_tomek = SMOTETomek(smote=SMOTE(k_neighbors=min(5, min_samples-1), random_state=42), random_state=42)
                X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_fold, y_train_fold)
                model.fit(
                    X_train_resampled, y_train_resampled,
                    eval_set=[(X_val_fold, y_val_fold)]
                )
            else:
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)]
                )
            # For classification, maximize ROC AUC
            y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
            fold_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        else:
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )
            # For regression, minimize RMSE
            y_val_pred = model.predict(X_val_fold)
            fold_score = -np.sqrt(mean_squared_error(y_val_fold, y_val_pred))  # Negative because Optuna minimizes
        
        cv_scores.append(fold_score)
    
    # If no valid folds were found (all folds had only one class)
    if len(cv_scores) == 0:
        raise ValueError("No valid folds found - all folds contain only one class. The dataset might be too imbalanced.")
    
    # Return mean CV score
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
    
    # Train final model with best parameters
    if task == 'classification':
        if model_name == 'lightgbm':
            final_model = LGBMClassifier(**best_params)
        elif model_name == 'xgboost':
            final_model = XGBClassifier(**best_params)
        elif model_name == 'catboost':
            final_model = CatBoostClassifier(**best_params)
        elif model_name == 'randomforest':
            final_model = RandomForestClassifier(**best_params)
    else:
        if model_name == 'lightgbm':
            final_model = LGBMRegressor(**best_params)
        elif model_name == 'xgboost':
            final_model = XGBRegressor(**best_params)
        elif model_name == 'catboost':
            final_model = CatBoostRegressor(**best_params)
        elif model_name == 'randomforest':
            final_model = RandomForestRegressor(**best_params)
    
    # Fit final model
    if task == 'classification':
        # Check if we have enough samples in each class for SMOTETomek
        y_train_int = y_train.astype(int)
        counts = np.bincount(y_train_int)
        min_samples = min(counts)
        
        if min_samples >= 3:
            # Apply SMOTETomek for class imbalance
            print("Applying SMOTETomek to balance training data...")
            smote_tomek = SMOTETomek(smote=SMOTE(k_neighbors=min(5, min_samples-1), random_state=42), random_state=42)
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
            final_model.fit(X_train_resampled, y_train_resampled)
        else:
            print("Not enough samples for SMOTETomek, training with original data...")
            final_model.fit(X_train, y_train)
    else:
        final_model.fit(X_train, y_train)
    
    # Save model
    model_output = (config.MODEL_OUTPUT_classification if task == 'classification' 
                   else config.MODEL_OUTPUT_regression)
    suffix = '_selected' if selected_features else ''
    model_path = os.path.join(
        model_output,
        f"{model_name}_{score_type.lower()}{suffix}.joblib"
    )
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Make predictions
    if task == 'classification':
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        optimal_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Generate and save plots
        plot_roc_curves(y_test, y_pred_proba, score_type, model_name, selected_features)
        # plot_precision_recall_curve(y_test, y_pred_proba, score_type, model_name, selected_features)
        # plot_confusion_matrix_custom(y_test, y_pred, score_type, model_name, selected_features)
        # plot_threshold_impact(y_test, y_pred_proba, score_type, model_name, selected_features)
        
        # Print classification metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    else:  # regression
        y_pred = final_model.predict(X_test)
        metrics = plot_regression_metrics(y_test, y_pred, score_type, model_name, selected_features)
        print("\nRegression Metrics:")
        print(f"RMSE: {metrics['rmse']:.3f}")
    
    # Plot and save feature importance
    plot_feature_importance(final_model, feature_names, score_type, model_name, task, selected_features)
    
    return final_model

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
                      choices=['lightgbm', 'xgboost', 'catboost', 'randomforest'],
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
