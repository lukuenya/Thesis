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
    confusion_matrix, classification_report, roc_curve, f1_score,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import config
import data_loader


# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS (same style as you had)
# -----------------------------------------------------------------------------

def plot_roc_curves(y_true, y_pred_proba, score_type, model_name, imputation=True, feature_selection=None):
    """Plot ROC curve for classification."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    
    # Get dynamic output path
    paths = config.get_output_paths(imputation, feature_selection)
    output_dir = os.path.join(paths['visualization'], model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        "roc_curve.png"
    ))
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, score_type, model_name, imputation=True, feature_selection=None):
    """Plot precision-recall curve for classification."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap_score = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap_score:.3f})')
    plt.axhline(y=np.sum(y_true) / len(y_true), color='navy', linestyle='--', label=f'Baseline (AP = {np.sum(y_true) / len(y_true):.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="upper right")
    
    # Get dynamic output path
    paths = config.get_output_paths(imputation, feature_selection)
    output_dir = os.path.join(paths['visualization'], model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        "pr_curve.png"
    ))
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred, score_type, model_name, imputation=True, feature_selection=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Get dynamic output path
    paths = config.get_output_paths(imputation, feature_selection)
    output_dir = os.path.join(paths['visualization'], model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        "cm.png"
    ))
    plt.close()


def plot_feature_importance(model, feature_names, score_type, model_name, task, imputation=True, feature_selection=None):
    """
    Plot feature importance for the trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        score_type: Score type ('FRIED', 'FRAGIRE18')
        model_name: Model name (e.g., 'lightgbm')
        task: Task type ('classification')
        imputation: Whether imputation was used
        feature_selection: Feature selection method used
    """
    # Get feature importances based on model type
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif model_name == 'lightgbm' and hasattr(model, 'feature_importance'):
        importances = model.feature_importance(importance_type='gain')
    else:
        print("Feature importance not available for this model")
        return
    
    # Create a DataFrame of feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance and get top 5
    importance_df = importance_df.sort_values('Importance', ascending=False).head(5)
    
    # Plot
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('')
    plt.tight_layout()
    
    # Get dynamic output paths
    paths = config.get_output_paths(imputation, feature_selection)
    output_dir = os.path.join(paths['visualization'], model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        f"feature_importance_{task}.png"
    ), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the feature importance to Excel
    feature_imp_dir = os.path.join(paths['feature_importances'])
    os.makedirs(feature_imp_dir, exist_ok=True)
    
    importance_df.to_excel(
        os.path.join(
            feature_imp_dir,
            f"{model_name}_{score_type.lower()}_feature_importances.xlsx"
        ),
        index=False
    )


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


def evaluate_model(model, X_test, y_test, score_type, model_name, imputation=True, feature_selection=None):
    """
    Evaluate a trained model and generate evaluation plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        score_type: Target score type (e.g., 'FRIED')
        model_name: Name of the model (e.g., 'lightgbm')
        imputation: Whether imputation was used
        feature_selection: Feature selection method used
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Generate and save plots
    plot_roc_curves(y_test, y_pred_proba, score_type, model_name, imputation, feature_selection)
    plot_precision_recall_curve(y_test, y_pred_proba, score_type, model_name, imputation, feature_selection)
    plot_confusion_matrix_custom(y_test, y_pred, score_type, model_name, imputation, feature_selection)
    
    if hasattr(model, 'feature_importances_') or (model_name == 'lightgbm' and hasattr(model, 'feature_importance')):
        plot_feature_importance(model, X_test.columns, score_type, model_name, 'classification', imputation, feature_selection)
    
    # Calculate and print metrics
    print("\nModel Evaluation:")
    print("================")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Print ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC: {roc_auc:.3f}")
    
    # Print average precision score
    ap_score = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision: {ap_score:.3f}")
    
    # Print F1 score
    f1 = f1_score(y_test, y_pred)
    print(f"F1 Score: {f1:.3f}")
    
    return {
        'roc_auc': roc_auc,
        'ap_score': ap_score,
        'f1': f1,
        'threshold': optimal_threshold
    }


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR OPTIMIZATION
# -----------------------------------------------------------------------------

def get_hyperparameter_space(trial, model_name):
    """Define hyperparameter space for the given model name."""
    if model_name == 'lightgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
    elif model_name == 'xgboost':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
        }
    elif model_name == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10.0),
            'border_count': trial.suggest_int('border_count', 10, 100)
        }
    elif model_name == 'randomforest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7, 0.5])
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def initialize_model(model_name, params):
    """Initialize model based on the given model name and parameters."""
    if model_name == 'lightgbm':
        return LGBMClassifier(**params, verbose=-1)
    elif model_name == 'xgboost':
        return XGBClassifier(**params, verbosity=0, use_label_encoder=False, eval_metric='logloss')
    elif model_name == 'catboost':
        return CatBoostClassifier(**params, verbose=0, thread_count=-1)
    elif model_name == 'randomforest':
        return RandomForestClassifier(**params, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def objective(trial, X, y, model_name):
    """Optuna objective function for hyperparameter optimization using Pipeline."""
    # Define hyperparameter space based on model name
    params = get_hyperparameter_space(trial, model_name)
    
    # Initialize model
    model = initialize_model(model_name, params)
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use SMOTETomek for class imbalance
    resampler = SMOTETomek(random_state=42)
    
    # Create pipeline with resampling
    pipeline = Pipeline([
        ('smote_tomek', resampler),
        ('model', model)
    ])
    
    # For each fold in CV, fit model and compute score
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Only apply resampling on training data to avoid data leakage
        if model_name in ['lightgbm', 'xgboost', 'catboost']:
            X_res, y_res = resampler.fit_resample(X_train_fold, y_train_fold)
            model.fit(X_res, y_res)
        else:
            pipeline.fit(X_train_fold, y_train_fold)
        
        # Get predictions
        if model_name in ['lightgbm', 'xgboost', 'catboost']:
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        else:
            y_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
        
        # Compute ROC AUC score
        roc_auc = roc_auc_score(y_val_fold, y_pred_proba)
        scores.append(roc_auc)
    
    # Return mean ROC AUC score across folds
    return np.mean(scores)


def optimize_model(score_type, n_trials=25, model_name='lightgbm', imputation=True, feature_selection=None):
    """
    Optimize model hyperparameters and train final model.
    
    Args:
        score_type: Target score type (e.g., 'FRIED', 'FRAGIRE18')
        n_trials: Number of trials for Optuna optimization
        model_name: Model to use ('lightgbm', 'xgboost', 'catboost', 'randomforest')
        imputation: Whether imputation was used
        feature_selection: Feature selection method used ('embedded', 'wrapper', or None)
    
    Returns:
        Dictionary with fitted model and evaluation results
    """
    print(f"Optimizing {model_name} for {score_type}")
    print(f"Imputation: {'Yes' if imputation else 'No'}")
    print(f"Feature Selection: {feature_selection if feature_selection else 'None'}")
    
    # Get data
    X, y = data_loader.load_data(
        target_score=score_type,
        imputation=imputation,
        feature_selection_method=feature_selection
    )
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize', study_name=f"{model_name}_{score_type}")
    study.optimize(partial(objective, X=X_train, y=y_train, model_name=model_name), n_trials=n_trials)
    
    # Get best hyperparameters
    best_params = study.best_params
    print(f"\nBest hyperparameters: {best_params}")
    
    # Initialize model with best hyperparameters
    best_model = initialize_model(model_name, best_params)
    
    # Use SMOTETomek for class imbalance in full training set
    resampler = SMOTETomek(random_state=42)
    X_train_res, y_train_res = resampler.fit_resample(X_train, y_train)
    
    # Fit model on resampled training data
    best_model.fit(X_train_res, y_train_res)
    
    # Evaluate model
    eval_results = evaluate_model(
        best_model, 
        X_test, 
        y_test, 
        score_type, 
        model_name,
        imputation,
        feature_selection
    )
    
    # Save model
    paths = config.get_output_paths(imputation, feature_selection)
    model_dir = os.path.join(paths['models'], model_name, score_type.upper())
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"{model_name}_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save hyperparameters
    hyperparam_path = os.path.join(model_dir, f"{model_name}_hyperparams.joblib")
    joblib.dump(best_params, hyperparam_path)
    print(f"Hyperparameters saved to: {hyperparam_path}")
    
    # Save evaluation results
    results_path = os.path.join(model_dir, f"{model_name}_eval_results.joblib")
    joblib.dump(eval_results, results_path)
    print(f"Evaluation results saved to: {results_path}")
    
    return {
        'model': best_model,
        'params': best_params,
        'eval_results': eval_results
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Optimize and train models')
    parser.add_argument('--score_type', type=str, default='FRIED',
                        help='Score to predict: FRIED or FRAGIRE18')
    parser.add_argument('--model', type=str, default='lightgbm',
                        help='Model to use: lightgbm, xgboost, catboost, randomforest')
    parser.add_argument('--n_trials', type=int, default=25,
                        help='Number of trials for Optuna optimization')
    parser.add_argument('--feature_selection', type=str, default=None,
                        help='Feature selection method: embedded, wrapper, or None')
    parser.add_argument('--no_imputation', action='store_true',
                        help='Use raw data without imputation')
    
    args = parser.parse_args()
    
    # Determine if imputation is used
    imputation = not args.no_imputation
    
    # Run optimization
    optimize_model(
        score_type=args.score_type,
        n_trials=args.n_trials,
        model_name=args.model,
        imputation=imputation,
        feature_selection=args.feature_selection
    )
