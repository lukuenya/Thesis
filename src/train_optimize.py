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

def plot_roc_curves(y_true, y_pred_proba, score_type, model_name, selected_features=False):
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
    
    suffix = '_selected' if selected_features else ''
    output_dir = os.path.join(config.VISUALIZATION_OUTPUT, model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        f"roc_curve{suffix}.png"
    ))
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, score_type, model_name, selected_features=False):
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
    
    suffix = '_selected' if selected_features else ''
    output_dir = os.path.join(config.VISUALIZATION_OUTPUT, model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        f"pr_curve{suffix}.png"
    ))
    plt.close()


def plot_confusion_matrix_custom(y_true, y_pred, score_type, model_name, selected_features=False):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    suffix = '_selected' if selected_features else ''
    output_dir = os.path.join(config.VISUALIZATION_OUTPUT, model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        f"cm{suffix}.png"
    ))
    plt.close()


def plot_feature_importance(model, feature_names, score_type, model_name, task, selected_features=False):
    """
    Plot feature importance for the trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        score_type: Score type ('FRIED', 'FRAGIRE18')
        model_name: Model name (e.g., 'lightgbm')
        task: Task type ('classification')
        selected_features: Whether selected features were used
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
    
    suffix = '_selected' if selected_features else ''
    output_dir = os.path.join(config.VISUALIZATION_OUTPUT, model_name, score_type.upper())
    os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(os.path.join(
        output_dir,
        f"feature_importance_{task}{suffix}.png"
    ), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the feature importance to Excel
    feature_imp_dir = os.path.join(config.MODEL_OUTPUT, 'feature_importances', 'feat_imp_classification')
    os.makedirs(feature_imp_dir, exist_ok=True)
    
    importance_df.to_excel(
        os.path.join(
            feature_imp_dir,
            f"{model_name}_{score_type.lower()}_feature_importances{suffix}.xlsx"
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


def evaluate_model(model, X_test, y_test, score_type, model_name, selected_features=False):
    """
    Evaluate a trained model and generate evaluation plots.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        score_type: Target score type (e.g., 'FRIED')
        model_name: Name of the model (e.g., 'lightgbm')
        selected_features: Whether selected features were used
    """
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Generate and save plots
    plot_roc_curves(y_test, y_pred_proba, score_type, model_name, selected_features)
    plot_precision_recall_curve(y_test, y_pred_proba, score_type, model_name, selected_features)
    plot_confusion_matrix_custom(y_test, y_pred, score_type, model_name, selected_features)
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and print additional metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # Plot and save feature importance
    if hasattr(model, 'feature_importances_') or hasattr(model, 'feature_importance'):
        feature_names = X_test.columns.tolist()
        plot_feature_importance(model, feature_names, score_type, model_name, 'classification', selected_features)


# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS FOR OPTIMIZATION
# -----------------------------------------------------------------------------

def get_hyperparameter_space(trial, model_name):
    """Define hyperparameter space for the given model name."""
    if model_name == 'randomforest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'random_state': 42
        }
    elif model_name == 'lightgbm':
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'rf']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbose': -1
        }
    elif model_name == 'xgboost':
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'verbosity': 0
        }
    elif model_name == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'auto_class_weights': 'Balanced',
            'verbose': False
        }
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return params


def initialize_model(model_name, params):
    """Initialize model based on the given model name and parameters."""
    if model_name == 'randomforest':
        return RandomForestClassifier(**params)
    elif model_name == 'lightgbm':
        return LGBMClassifier(**params)
    elif model_name == 'xgboost':
        return XGBClassifier(**params)
    elif model_name == 'catboost':
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def objective(trial, X, y, model_name):
    """Optuna objective function for hyperparameter optimization using Pipeline."""
    # Define hyperparameter space based on model
    params = get_hyperparameter_space(trial, model_name)
    
    # Initialize model with current hyperparameters
    model = initialize_model(model_name, params)
    
    # Count samples in each class to determine SMOTE parameters
    if isinstance(y, pd.Series):
        y_int = y.astype(int)
    else:
        y_int = y.astype(int)
    
    counts = np.bincount(y_int)
    min_samples = min(counts)
    k_neighbors = min(5, min_samples-1) if min_samples >= 3 else 1
    
    # Use 3-fold cross-validation (because of limited data in positive class)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    skipped_folds = 0
    
    # Pipeline with SMOTETomek for handling class imbalance
    pipeline = Pipeline([
        ('sampling', SMOTETomek(
            smote=SMOTE(k_neighbors=k_neighbors, random_state=42), 
            random_state=42
        ) if min_samples >= 3 else None),
        ('model', model)
    ])
    
    # If we don't have enough samples for SMOTE, remove the sampling step
    if min_samples < 3:
        pipeline.steps.pop(0)
    
    # Cross-validation
    for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        try:
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
                skipped_folds += 1
                continue  # Skip this fold
            
            # Fit the pipeline on training data
            if model_name == 'randomforest':
                pipeline.fit(X_train_fold, y_train_fold)
            else:
                # For models that support eval_set
                # Need to fit model directly after sampling
                X_train_resampled, y_train_resampled = X_train_fold, y_train_fold
                
                # Apply sampling if it's in the pipeline
                if min_samples >= 3:
                    X_train_resampled, y_train_resampled = pipeline.named_steps['sampling'].fit_resample(
                        X_train_fold, y_train_fold
                    )
                
                model.fit(
                    X_train_resampled, y_train_resampled,
                    eval_set=[(X_val_fold, y_val_fold)]
                )
            
            # Evaluate on validation fold
            if model_name == 'randomforest':
                y_val_pred_proba = pipeline.predict_proba(X_val_fold)[:, 1]
            else:
                y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
                
            fold_score = roc_auc_score(y_val_fold, y_val_pred_proba)
            cv_scores.append(fold_score)
        except Exception as e:
            print(f"Error in fold {i}: {str(e)}")
            continue
    
    # Log if too many folds are being skipped
    if skipped_folds > 0:
        print(f"Warning: Skipped {skipped_folds}/{cv.n_splits} folds due to single-class validation sets")
    
    # Return mean score across folds
    mean_score = np.mean(cv_scores) if cv_scores else 0.0
    return mean_score


def optimize_model(score_type, n_trials=25, model_name='lightgbm', selected_features=False):
    """Optimize model hyperparameters and train final model."""
    print(f"Loading data for {score_type}...")
    X, y = data_loader.load_data(
        target_score=score_type,
        selected_features=selected_features
    )
    
    # Create output directories if they don't exist
    os.makedirs(config.MODEL_OUTPUT, exist_ok=True)
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    
    # Create model and score specific visualization directory
    model_score_vis_dir = os.path.join(config.VISUALIZATION_OUTPUT, model_name, score_type.upper())
    os.makedirs(model_score_vis_dir, exist_ok=True)
    
    # Run hyperparameter optimization with cross-validation
    study = optuna.create_study(direction='maximize')
    objective_func = lambda trial: objective(trial, X, y, model_name)
    study.optimize(objective_func, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best score: {study.best_value}")
    
    # Calculate cross-validation scores with standard deviation
    cv_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize model with best parameters
    best_model = initialize_model(model_name, best_params)
    
    # Perform cross-validation to get mean and standard deviation
    for train_idx, val_idx in cv.split(X, y):
        # Handle DataFrame/Series indexing
        if isinstance(X, pd.DataFrame):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            
        if isinstance(y, pd.Series):
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Skip folds with only one class
        if len(np.unique(y_val_fold)) == 1:
            continue
        
        # Create and fit pipeline with best parameters
        y_int = y_train_fold.astype(int)
        counts = np.bincount(y_int)
        min_samples = min(counts)
        k_neighbors = min(5, min_samples-1) if min_samples >= 3 else 1
        
        if min_samples >= 3:
            smote_tomek = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=42), random_state=42)
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_fold, y_train_fold)
            best_model.fit(X_train_resampled, y_train_resampled)
        else:
            best_model.fit(X_train_fold, y_train_fold)
        
        # Calculate ROC AUC
        y_val_pred_proba = best_model.predict_proba(X_val_fold)[:, 1]
        fold_score = roc_auc_score(y_val_fold, y_val_pred_proba)
        cv_scores.append(fold_score)
    
    # Calculate mean and standard deviation of CV scores
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"Cross-validation ROC AUC: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Check if we have enough samples for SMOTETomek
    y_int = y_train.astype(int)
    counts = np.bincount(y_int)
    min_samples = min(counts)
    k_neighbors = min(5, min_samples-1) if min_samples >= 3 else 1
    
    # Initialize model with best parameters
    final_model = initialize_model(model_name, study.best_params)
    
    # Create and fit pipeline for final model
    if min_samples >= 3:
        print("Applying SMOTETomek for final model training...")
        pipeline = Pipeline([
            ('sampling', SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=42), random_state=42)),
            ('model', final_model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # For compatibility with the rest of the code, extract the model
        final_model = pipeline.named_steps['model']
    else:
        print("Not enough samples for SMOTETomek, training model directly...")
        final_model.fit(X_train, y_train)
    
    # Save model
    model_output = config.MODEL_OUTPUT
    suffix = '_selected' if selected_features else ''
    model_path = os.path.join(
        model_output,
        f"{model_name}_{score_type.lower()}{suffix}.pkl"
    )
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate model
    evaluate_model(final_model, X_test, y_test, score_type, model_name, selected_features)
    
    return final_model, X_test, y_test


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
