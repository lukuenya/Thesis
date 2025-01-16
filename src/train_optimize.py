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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT,
        f'roc_curve_{score_type.lower()}_{model_name}{suffix}.png'
    ))
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
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT,
        f'pr_curve_{score_type.lower()}_{model_name}{suffix}.png'
    ))
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
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT,
        f'confusion_matrix_{score_type.lower()}_{model_name}{suffix}.png'
    ))
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
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT,
        f'threshold_impact_{score_type.lower()}_{model_name}{suffix}.png'
    ))
    plt.close()
    
    return optimal_threshold, optimal_f1



def plot_feature_importance(feature_importances, feature_names, score_type, model_name, selected_features=False):
    """Plot feature importance and save to file."""
    # Convert to DataFrame for easy plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    # Select only top 20 features
    df = df.head(20)
    
    # Plot with larger figure size and horizontal bars
    plt.figure(figsize=(12, 10))
    sns.barplot(data=df, x='Importance', y='Feature')
    plt.title(f'Top 20 Features - {score_type} ({model_name})')
    plt.xlabel('Importance')
    
    # Rotate feature names for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(
        config.VISUALIZATION_OUTPUT,
        f'feature_importance_{score_type.lower()}_{model_name}{suffix}.png'
    ), bbox_inches='tight', dpi=300)
    plt.close()


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
            'n_estimators': trial.suggest_int('n_estimators', 30, 200),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
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


def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold that maximizes F1 score."""
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


def objective(trial, X, y, model_name):
    """Optuna objective function for hyperparameter optimization via CV."""
    params = get_hyperparameter_space(trial, model_name)
    
    # Create model
    if model_name == "lightgbm":
        model = LGBMClassifier(**params)
    elif model_name == "xgboost":
        model = XGBClassifier(**params)
    elif model_name == "catboost":
        model = CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform cross-validation
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Get probabilities and find optimal threshold
        y_val_proba = model.predict_proba(X_val_fold)[:, 1]
        
        # Calculate AUC for this fold
        fold_auc = roc_auc_score(y_val_fold, y_val_proba)
        cv_scores.append(fold_auc)
    
    # Return mean AUC across folds
    mean_auc = np.mean(cv_scores)
    return mean_auc


# -----------------------------------------------------------------------------
# MAIN TRAINING + OPTIMIZATION + PLOTTING
# -----------------------------------------------------------------------------

def optimize_model(score_type, n_trials, model_name, selected_features=False):
    """
    1) Loads data (with or without selected features).
    2) Splits into train/test.
    3) Runs Optuna CV on train only.
    4) Retrains on entire train set with best params.
    5) Evaluates on hold-out test set.
    6) Saves & plots everything (ROC, PR, confusion matrix, etc.).
    7) Saves final model + feature importances.
    """
    
    # 1. Load the dataset
    if selected_features:
        X_full, y_full, feature_names = data_loader.load_data_with_selected_features(score_type)
    else:
        X_full, y_full, feature_names = data_loader.load_data(score_type)
    
    print(f"Loaded dataset with shape: {X_full.shape}")
    print(f"Number of features: {len(feature_names)}")

    # 2. Split off a test set (unseen data)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, 
        test_size=0.25,
        random_state=42,
        stratify=y_full
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 3. Hyperparameter optimization with Optuna
    optimization_fn = partial(objective, X=X_train, y=y_train, model_name=model_name)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimization_fn, n_trials=n_trials)

    best_params = study.best_params
    best_cv_auc = study.best_value
    print(f"\n[Optuna] Best CV AUC: {best_cv_auc:.4f}")
    print(f"[Optuna] Best params: {best_params}")

    # 4. Retrain final model on all training data
    if model_name == "lightgbm":
        final_model = LGBMClassifier(**best_params)
    elif model_name == "xgboost":
        final_model = XGBClassifier(**best_params)
    elif model_name == "catboost":
        final_model = CatBoostClassifier(**best_params)

    final_model.fit(X_train, y_train)

    # 5. Evaluate on test set
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)
    print(f"\nHold-out Test AUC: {test_auc:.4f}")

    # (Optionally find threshold based on training data)
    train_proba = final_model.predict_proba(X_train)[:, 1]
    best_thresh, best_f1 = find_optimal_threshold(y_train, train_proba)
    print(f"Optimal threshold on training set = {best_thresh:.2f} (F1={best_f1:.2f})")

    # Predictions on test set with that threshold
    y_test_pred = (test_proba > best_thresh).astype(int)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))

    # 6. Create directory for plots if not exists
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)

    # Plot on test set
    plot_roc_curves(y_test, test_proba, score_type, model_name, selected_features)
    plot_precision_recall_curve(y_test, test_proba, score_type, model_name, selected_features)
    plot_confusion_matrix_custom(y_test, y_test_pred, score_type, model_name, selected_features)
    plot_threshold_impact(y_test, test_proba, score_type, model_name, selected_features)

    # 7. Save final model + metrics
    # Also get feature importances if supported by the model
    model_filename = (
        f"{model_name}_{score_type.lower()}"
        + ("_selected_features" if selected_features else "_final")
        + ".joblib"
    )

    save_dict = {
        'model': final_model,
        'optimal_threshold': best_thresh,
        'feature_names': feature_names,
        'best_params': best_params,
        'cv_best_auc': best_cv_auc,
        'holdout_test_auc': test_auc
    }
    joblib.dump(save_dict, os.path.join(config.MODEL_OUTPUT, model_filename))
    print(f"\nModel saved to {model_filename}")

    # 8. Save feature importances (if the model provides them)
    if hasattr(final_model, 'feature_importances_'):
        feature_importances = final_model.feature_importances_
        feature_importances_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        # Save .xlsx
        importance_filename = (
            f"{model_name}_{score_type.lower()}"
            + ("_selected_features_importance.xlsx" if selected_features else "_final_importance.xlsx")
        )
        feature_importances_df.to_excel(
            os.path.join(config.MODEL_OUTPUT, importance_filename),
            index=False
        )
        print(f"Feature importances saved to {importance_filename}")

        # Also plot feature importances
        plot_feature_importance(feature_importances, feature_names, score_type, model_name, selected_features)


# -----------------------------------------------------------------------------
# CLI ENTRY
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train, Optimize, and Plot Model Performance')
    parser.add_argument('--score_type', type=str, default='FRIED',
                        choices=['FRIED', 'FRAGIRE18'],
                        help='Which label or score to optimize for.')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials.')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                        choices=['lightgbm', 'xgboost', 'catboost'],
                        help='Which model to optimize.')
    parser.add_argument('--selected_features', action='store_true',
                        help='Whether to use only selected features from data_loader.')

    args = parser.parse_args()
    optimize_model(
        score_type=args.score_type,
        n_trials=args.n_trials,
        model_name=args.model_name,
        selected_features=args.selected_features
    )
