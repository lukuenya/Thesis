import os
import numpy as np
import pandas as pd
import joblib
import optuna
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_curve, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import config
import data_loader


# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (same style as you had)
# -----------------------------------------------------------------------------

def plot_roc_curves(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot ROC curves."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6), dpi=300)  # High resolution
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f"roc_curve_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6), dpi=300)  # High resolution
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.axhline(y=np.sum(y_true) / len(y_true), color='navy', linestyle='--', label=f'Baseline (AP = {np.sum(y_true) / len(y_true):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f"pr_curve_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred, score_type, model_name, selected_features=False):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)  # High resolution
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT_Classification,
        f"cm_{model_name}_{score_type.lower()}{suffix}.png"
    ))
    plt.close()


def plot_feature_importance(model, feature_names, score_type, model_name, task, selected_features=False):
    """Plot feature importance for model."""
    if model_name == 'lightgbm':
        importance = model.feature_importances_
        indices = np.argsort(importance)
    elif model_name == 'xgboost':
        importance = model.feature_importances_
        indices = np.argsort(importance)
    elif model_name == 'catboost':
        importance = model.feature_importances_
        indices = np.argsort(importance)
    elif model_name == 'randomforest':
        importance = model.feature_importances_
        indices = np.argsort(importance)
    else:
        return
    
    plt.figure(figsize=(12, len(feature_names) * 0.5), dpi=300)  # High resolution
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    output_dir = config.VISUALIZATION_OUTPUT_Classification
    plt.savefig(os.path.join(
        output_dir,
        f"feature_importance_{model_name}_{score_type.lower()}_{task}{suffix}.png"
    ))
    plt.close()

    # Also save to CSV
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    feature_importance_df = feature_importance_df.sort_values(
        by='Importance', ascending=False
    )
    
    feature_importance_df.to_csv(os.path.join(
        output_dir,
        f"feature_importance_{model_name}_{score_type.lower()}_{task}{suffix}.csv"
    ), index=False)


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal threshold for classification."""
    thresholds = np.arange(0, 1, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        else:
            score = roc_auc_score(y_true, y_pred_proba)
        scores.append(score)
    
    best_score_idx = np.argmax(scores)
    best_threshold = thresholds[best_score_idx]
    best_score = scores[best_score_idx]
    
    return best_threshold, best_score


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR OPTIMIZATION
# -----------------------------------------------------------------------------

def get_hyperparameter_space(trial, model_name):
    """Define hyperparameter search space for each model."""
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

            'n_estimators': trial.suggest_int('n_estimators', 300, 500),
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

            'iterations': trial.suggest_int('iterations', 300, 500),
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


def initialize_model(model_name, params):
    """Initialize model based on model name."""
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


def objective(trial, X, y, model_name):
    """Optuna objective function for hyperparameter optimization."""
    # Define hyperparameter space based on model
    params = get_hyperparameter_space(trial, model_name)
    
    # Define cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    # Convert y to integers for SMOTETomek
    if isinstance(y, pd.Series):
        y_int = y.astype(int)
    else:
        y_int = y.astype(int)
    
    # Count samples in each class
    counts = np.bincount(y_int)
    min_samples = min(counts)
    
    # Cross-validation
    for train_idx, val_idx in cv.split(X, y):
        # Convert to DataFrame and Series if they're not already
        if isinstance(X, pd.DataFrame):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            
        if isinstance(y, pd.Series):
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Check if validation fold has only one class
        if len(np.unique(y_val_fold)) == 1:
            continue  # Skip this fold
        
        # Initialize model with current hyperparameters
        model = initialize_model(model_name, params)
        
        # Apply SMOTETomek for class imbalance if we have enough samples
        if min_samples >= 3:
            # Use SMOTETomek (SMOTE + Tomek links)
            smote_tomek = SMOTETomek(smote=SMOTE(k_neighbors=min(5, min_samples-1), random_state=42), random_state=42)
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_fold, y_train_fold)
            
            # For tree-based models that don't have eval_set parameter
            if model_name == 'randomforest':
                model.fit(X_train_resampled, y_train_resampled)
            else:
                model.fit(
                    X_train_resampled, y_train_resampled,
                    eval_set=[(X_val_fold, y_val_fold)]
                )
        else:
            # For tree-based models that don't have eval_set parameter
            if model_name == 'randomforest':
                model.fit(X_train_fold, y_train_fold)
            else:
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)]
                )
        # For classification, maximize ROC AUC
        y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        fold_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        
        cv_scores.append(fold_score)
    
    # Return mean score across folds
    mean_score = np.mean(cv_scores) if cv_scores else 0.0
    return mean_score


def optimize_model(score_type, n_trials=25, model_name='lightgbm', selected_features=False):
    """Optimize model for specified score type."""
    print(f"\nOptimizing {model_name} for {score_type}...")
    
    # Load data
    X, y = data_loader.load_data(score_type, selected_features)
    
    # Split data into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Create output directories if they don't exist
    os.makedirs(config.MODEL_OUTPUT, exist_ok=True)
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    
    # Run hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    objective_func = partial(objective, X=X_train, y=y_train, model_name=model_name)
    study.optimize(objective_func, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    print(f"\nBest parameters: {best_params}")
    
    # Train final model with best parameters
    if model_name == 'lightgbm':
        final_model = LGBMClassifier(**best_params)
    elif model_name == 'xgboost':
        final_model = XGBClassifier(**best_params)
    elif model_name == 'catboost':
        final_model = CatBoostClassifier(**best_params)
    elif model_name == 'randomforest':
        final_model = RandomForestClassifier(**best_params)
    
    # Fit final model
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
    
    # Save model
    model_output = config.MODEL_OUTPUT
    suffix = '_selected' if selected_features else ''
    model_path = os.path.join(
        model_output,
        f"{model_name}_{score_type.lower()}{suffix}.joblib"
    )
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Make predictions
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Generate and save plots
    plot_roc_curves(y_test, y_pred_proba, score_type, model_name, selected_features)
    plot_precision_recall_curve(y_test, y_pred_proba, score_type, model_name, selected_features)
    plot_confusion_matrix_custom(y_test, y_pred, score_type, model_name, selected_features)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(classification_report(y_test, y_pred))
    
    # Plot and save feature importance
    feature_names = X.columns.tolist()
    plot_feature_importance(final_model, feature_names, score_type, model_name, 'classification', selected_features)
    
    return final_model


# -----------------------------------------------------------------------------
# CLI ENTRY
# -----------------------------------------------------------------------------

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
