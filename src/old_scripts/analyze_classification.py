import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    roc_auc_score, f1_score
)
import config
import data_loader
from sklearn.model_selection import train_test_split

def plot_roc_curves(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot ROC curve"""
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
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'roc_curve_{score_type.lower()}_{model_name}{suffix}.png'))
    plt.close()

def plot_precision_recall_curve(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot Precision-Recall curve"""
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
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'pr_curve_{score_type.lower()}_{model_name}{suffix}.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, score_type, model_name, selected_features=False):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {score_type} ({model_name})')
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'confusion_matrix_{score_type.lower()}_{model_name}{suffix}.png'))
    plt.close()

def plot_feature_importance(feature_importance_df, score_type, model_name, selected_features=False, top_n=20):
    """Plot feature importances"""
    # Sort and get top N features
    df = feature_importance_df.sort_values('Importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Importance', y='Feature')
    plt.title(f'Top {top_n} Feature Importance - {score_type} ({model_name})')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'feature_importance_{score_type.lower()}_{model_name}{suffix}.png'))
    plt.close()

def plot_threshold_impact(y_true, y_pred_proba, score_type, model_name, selected_features=False):
    """Plot impact of different thresholds on F1 score"""
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
    
    # Find and plot optimal threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    plt.plot(optimal_threshold, optimal_f1, 'r*', markersize=15,
             label=f'Optimal Threshold = {optimal_threshold:.2f}\nF1 Score = {optimal_f1:.2f}')
    plt.legend()
    
    suffix = '_selected' if selected_features else ''
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'threshold_impact_{score_type.lower()}_{model_name}{suffix}.png'))
    plt.close()
    
    return optimal_threshold, optimal_f1

def analyze_model(score_type, model_name, selected_features=False):
    """Analyze a trained model's performance"""
    # Determine model filename based on whether it uses selected features
    suffix = '_selected_features' if selected_features else '_optimized'
    model_filename = f"{model_name}_{score_type.lower()}{suffix}.joblib"
    model_path = os.path.join(config.MODEL_OUTPUT, model_filename)
    
    try:
        loaded_obj = joblib.load(model_path)
        # Handle both old format (just model) and new format (dictionary)
        if isinstance(loaded_obj, dict):
            model = loaded_obj['model']
            optimal_threshold = loaded_obj.get('optimal_threshold', 0.3)
        else:
            model = loaded_obj
            optimal_threshold = 0.3
    except FileNotFoundError:
        print(f"No model found at {model_path}")
        return
    
    # Load data
    if selected_features:
        X, y, feature_names = data_loader.load_data_with_selected_features(score_type)
    else:
        X, y, feature_names = data_loader.load_data(score_type)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    
    # Create visualizations directory if it doesn't exist
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    
    # Generate plots
    plot_roc_curves(y, y_pred_proba, score_type, model_name, selected_features)
    plot_precision_recall_curve(y, y_pred_proba, score_type, model_name, selected_features)
    plot_confusion_matrix(y, y_pred, score_type, model_name, selected_features)
    
    # Load and plot feature importances
    importance_file = f"{model_name}_{score_type.lower()}{suffix}_importance.xlsx"
    try:
        feature_importance_df = pd.read_excel(os.path.join(config.MODEL_OUTPUT, importance_file))
        plot_feature_importance(feature_importance_df, score_type, model_name, selected_features)
    except FileNotFoundError:
        print(f"No feature importance file found for {model_name}")
    
    # Print classification report
    print(f"\nClassification Report for {model_name} on {score_type} (Threshold = {optimal_threshold:.2f}):")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze model performance')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18'],
                      help='Which score type to analyze')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost'],
                      help='Which model to analyze')
    parser.add_argument('--selected_features', action='store_true',
                      help='Whether to analyze model with selected features')
    
    args = parser.parse_args()
    analyze_model(args.score_type, args.model_name, args.selected_features)
