import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import config
import re


def clean_name(name):
    """Clean a single column name to be compatible with LightGBM"""
    cleaned = re.sub(r'[^A-Za-z0-9_]+', '_', str(name))
    cleaned = cleaned.strip('_')
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned


def clean_column_names(df):
    """Clean column names to be compatible with LightGBM"""
    df.columns = [clean_name(col) for col in df.columns]
    return df


def analyze_model_performance(score_type='FRIED'):
    """Analyze model performance and generate comprehensive visualizations"""
    # Load data
    df = pd.read_csv(config.TRAINING_FILE_FOLD)
    df = clean_column_names(df)
    
    # Clean target names
    fried_target = clean_name('Fried_Score_FRIED_TOTAL_Version_1')
    fragire_target = clean_name('Frailty_Score_FRAGIRE18_SQ001')
    
    # Update config column names
    cleaned_fried_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRIED_SCORE]
    cleaned_fragire_cols = [clean_name(col) for col in config.COLS_TO_DROP_FRAILTY_SCORE]
    
    # Drop rows with NaN in target
    if score_type == 'FRIED':
        df = df.dropna(subset=[fried_target])
        target = fried_target
        cols_to_drop = cleaned_fried_cols + [fragire_target, 'kfold']
        model_path = os.path.join(config.MODEL_OUTPUT, "lightgbm_fried_optimized.joblib")
    else:
        df = df.dropna(subset=[fragire_target])
        target = fragire_target
        cols_to_drop = cleaned_fragire_cols + [fried_target, 'kfold']
        model_path = os.path.join(config.MODEL_OUTPUT, "lightgbm_fragire18_optimized.joblib")
    
    # Prepare features
    features = df.drop(cols_to_drop, axis=1)
    X = features.select_dtypes(include=[np.number])
    y = df[target]
    
    # Load model
    model = joblib.load(model_path)
    
    # Generate predictions
    y_pred = model.predict(X)
    errors = y_pred - y
    
    # Calculate and print metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    print(f"\nModel Performance for {score_type} Score:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"\nError Statistics:")
    print(f"Mean Error: {errors.mean():.4f}")
    print(f"Std Error: {errors.std():.4f}")
    print(f"Error Range: [{errors.min():.4f}, {errors.max():.4f}]")
    
    # Create visualizations directory if it doesn't exist
    viz_dir = os.path.join(config.MODEL_OUTPUT, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title(f'Actual vs Predicted {score_type} Scores')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{score_type.lower()}_prediction_scatter.png'))
    plt.close()
    
    # 2. Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=30, kde=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(f'Error Distribution - {score_type} Score')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{score_type.lower()}_error_distribution.png'))
    plt.close()
    
    # 3. Feature Importance
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(20), x='importance', y='feature')
    plt.title(f'Top 20 Feature Importance - {score_type} Score')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'{score_type.lower()}_feature_importance.png'))
    plt.close()
    
    # Save feature importance to CSV
    importance_df.to_csv(
        os.path.join(viz_dir, f'{score_type.lower()}_feature_importance.csv'),
        index=False
    )
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': model,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'importance_df': importance_df,
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'error_mean': errors.mean(),
            'error_std': errors.std()
        }
    }


if __name__ == "__main__":
    # Analyze both models
    print("Analyzing FRIED Score Model...")
    fried_results = analyze_model_performance('FRIED')
    
    print("\nAnalyzing FRAGIRE18 Score Model...")
    fragire_results = analyze_model_performance('FRAGIRE18')
