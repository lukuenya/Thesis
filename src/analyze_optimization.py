# src/analyze_optimization.py

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import config

def plot_learning_curves(model, X_train, y_train, X_test, y_test, score_type, model_name):
    """Plot learning curves to check for overfitting/underfitting"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_rmse = []
    test_rmse = []
    
    for size in train_sizes:
        # Take a subset of training data
        n_samples = int(len(X_train) * size)
        X_subset = X_train.iloc[:n_samples]
        y_subset = y_train[:n_samples]
        
        # Train model on subset
        model.fit(X_subset, y_subset)
        
        # Calculate RMSE on training and test sets
        train_pred = model.predict(X_subset)
        test_pred = model.predict(X_test)
        
        train_rmse.append(np.sqrt(mean_squared_error(y_subset, train_pred)))
        test_rmse.append(np.sqrt(mean_squared_error(y_test, test_pred)))
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_rmse, label='Training RMSE', marker='o')
    plt.plot(train_sizes, test_rmse, label='Test RMSE', marker='o')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('RMSE')
    plt.title(f'Learning Curves - {score_type} Score ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'learning_curves_{score_type.lower()}_{model_name}.png'))
    plt.close()

def plot_predictions(y_test, y_pred, score_type, model_name):
    """Create scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', label='Perfect Prediction')
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predicted vs Actual - {score_type} Score ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'predictions_{score_type.lower()}_{model_name}.png'))
    plt.close()

def plot_feature_importance(model, feature_names, score_type, model_name):
    """Plot top 10 feature importances"""
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return  # Skip if model doesn't support feature importance
    
    # Create dataframe of feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(10)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Top 10 Feature Importance - {score_type} Score ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(config.VISUALIZATION_OUTPUT, 
                            f'feature_importance_{score_type.lower()}_{model_name}.png'))
    plt.close()

def analyze_model(score_type, model_name):
    """Analyze a single model's performance"""
    # Load model results
    model_filename = f"{model_name}_{score_type.lower()}_optimized.joblib"
    try:
        results = joblib.load(os.path.join(config.MODEL_OUTPUT, model_filename))
    except FileNotFoundError:
        print(f"No results found for {model_name} on {score_type} score")
        return
    
    # Extract data
    model = results['model']
    y_test = results['y_test']
    y_pred = results['y_pred']
    test_indices = results['test_indices']
    
    # Load original data for feature names
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    
    # Get target column and columns to drop
    target_col = ('Fried_Score_FRIED_TOTAL_Version_1' if score_type == 'FRIED' 
                 else 'Frailty_Score_FRAGIRE18_SQ001')
    cols_to_drop = (config.COLS_TO_DROP_FRIED_SCORE if score_type == 'FRIED'
                   else config.COLS_TO_DROP_FRAILTY_SCORE)
    
    # Prepare data
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    X = df.drop(cols_to_drop + [target_col], axis=1)
    y = df[target_col].values
    
    # Get train/test split
    fold_col = f'kfold_{score_type.lower()}'
    train_idx = df[~df[fold_col].isin([test_indices[0]])].index
    
    X_train = X.loc[train_idx]
    y_train = y[train_idx]
    X_test = X.loc[test_indices]
    y_test = y[test_indices]
    
    # Create visualizations
    plot_learning_curves(model, X_train, y_train, X_test, y_test, score_type, model_name)
    plot_predictions(y_test, y_pred, score_type, model_name)
    plot_feature_importance(model, X.columns, score_type, model_name)

if __name__ == "__main__":
    # Create visualization directory
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    
    # Models to analyze
    models = ["lightgbm", "xgboost", "catboost"]
    
    # Analyze each model for both score types
    for model_name in models:
        try:
            print(f"\nAnalyzing {model_name}...")
            analyze_model('FRIED', model_name)
            analyze_model('FRAGIRE18', model_name)
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue

