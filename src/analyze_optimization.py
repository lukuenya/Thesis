import optuna
import joblib
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import learning_curve
import data_loader
import config
import os
import re
import pandas as pd


def clean_name(name):
    """Clean a single column name to be compatible with LightGBM"""
    # Replace special characters and spaces with underscore
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', str(name))
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Replace multiple underscores with single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned


def clean_column_names(df):
    """Clean column names to be compatible with LightGBM"""
    df.columns = [clean_name(col) for col in df.columns]
    return df


def plot_learning_curves(model, X, y, score_type, title, save_path=None):
    """Plot learning curves for a trained model"""
    # Convert to numpy arrays if they're pandas objects
    X = X.to_numpy() if hasattr(X, 'to_numpy') else X
    y = y.to_numpy() if hasattr(y, 'to_numpy') else y
    
    # Load the fold indices for cross-validation
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    
    # Get target column name and fold column
    target_col = 'Fried_Score_FRIED_TOTAL_Version_1' if score_type == 'FRIED' else 'Frailty_Score_FRAGIRE18_SQ001'
    fold_col = f'kfold_{score_type.lower()}'
    
    # Drop NaN values and get valid indices that match our X and y
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    # Create CV folds from the cleaned data
    cv_folds = [(df[df[fold_col] != i].index, df[df[fold_col] == i].index) for i in range(5)]
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=cv_folds,  # Use our pre-defined folds
        scoring='neg_mean_squared_error',  
        n_jobs=-1,
        random_state=42
    )
    
    # Calculate RMSE from MSE
    train_rmse = np.sqrt(-train_scores)  
    val_rmse = np.sqrt(-val_scores)
    
    # Calculate mean and std
    train_mean = np.mean(train_rmse, axis=1)
    train_std = np.std(train_rmse, axis=1)
    val_mean = np.mean(val_rmse, axis=1)
    val_std = np.std(val_rmse, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color='blue'
    )
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='orange')
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.1,
        color='orange'
    )
    
    plt.xlabel('Training Examples')
    plt.ylabel('RMSE')
    plt.title(f'Learning Curves - {title}')
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names, score_type, viz_dir):
    """Plot feature importance and return the most important features"""
    # Get feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance['importance'])
    plt.xticks(range(len(importance)), importance['feature'], rotation=90)
    plt.title(f'Feature Importance - {score_type} Score')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{score_type.lower()}_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Select top features (those that account for 95% of total importance)
    total_importance = importance['importance'].sum()
    cumulative_importance = importance['importance'].cumsum() / total_importance
    important_features = importance[cumulative_importance <= 0.95]['feature'].tolist()
    
    print(f"\nFeature Importance Analysis for {score_type}:")
    print(f"Total features: {len(feature_names)}")
    print(f"Selected features: {len(important_features)} (95% of total importance)")
    print("\nTop 10 most important features:")
    for idx, row in importance.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f} ({row['importance'] / total_importance * 100:.2f}%)")
    
    return important_features


def analyze_optimization(score_type):
    """Analyze optimization results for a specific score type"""
    # Create visualization directory if it doesn't exist
    viz_dir = os.path.join(config.MODEL_OUTPUT, 'visualizations', 'optimization')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load study and model
    study_filename = f"{score_type.lower()}_study.joblib"
    if score_type == 'FRAGIRE18':
        study_filename = "fragire18_study.joblib"
    
    study = joblib.load(os.path.join(config.MODEL_OUTPUT, study_filename))
    model = joblib.load(os.path.join(config.MODEL_OUTPUT, f"lightgbm_{score_type.lower()}_optimized.joblib"))
    
    print(f"\nOptimization Analysis for {score_type}:")
    print("Best value (RMSE):", study.best_value)
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    
    # Load data for learning curves
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    df = clean_column_names(df)
    
    # Select appropriate target column and columns to drop
    target_col = 'Fried_Score_FRIED_TOTAL_Version_1' if score_type == 'FRIED' else 'Frailty_Score_FRAGIRE18_SQ001'
    cols_to_drop = config.COLS_TO_DROP_FRIED_SCORE if score_type == 'FRIED' else config.COLS_TO_DROP_FRAILTY_SCORE
    cols_to_drop = [clean_name(col) for col in cols_to_drop]
    
    # Drop rows with NaN in target first
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    # Drop unnecessary columns including both fold columns
    cols_to_drop_all = cols_to_drop + ['kfold_fried', 'kfold_fragire18']
    X = df.drop(cols_to_drop_all + [target_col], axis=1)
    y = df[target_col]
    
    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Get feature names from the model
    if hasattr(model, 'feature_name_'):
        feature_names = model.feature_name_
    else:
        # If feature names not stored, use the ones that match the number of features
        n_features = len(model.feature_importances_)
        feature_names = X.columns[:n_features]
    
    # Plot feature importance using only the features used by the model
    important_features = plot_feature_importance(model, feature_names, score_type, viz_dir)
    
    # Filter X to only include important features
    X_important = X[important_features]
    
    print("\nRetraining model with selected features...")
    
    # Plot learning curves with selected features
    plot_learning_curves(
        model, X_important, y,
        score_type=score_type,
        title=f'{score_type} Score (Selected Features)',
        save_path=os.path.join(viz_dir, f'{score_type.lower()}_learning_curves_selected.png')
    )
    
    # Plot and save visualizations
    # 1. Optimization History
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(os.path.join(viz_dir, f'{score_type.lower()}_optimization_history.html'))
    
    # 2. Parameter Importance
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(os.path.join(viz_dir, f'{score_type.lower()}_param_importance.html'))
    
    # 3. Parallel Coordinate
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_html(os.path.join(viz_dir, f'{score_type.lower()}_parallel_coordinate.html'))


if __name__ == "__main__":
    # Analyze both score types
    analyze_optimization('FRIED')
    analyze_optimization('FRAGIRE18')
