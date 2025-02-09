import pandas as pd
import numpy as np
import joblib
import os
import data_loader
import config
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_importances(score_type, task='classification'):
    """
    Load feature importance files for all models
    """
    importances = {}
    models = ['lightgbm', 'xgboost', 'catboost']
    
    for model in models:
        filepath = os.path.join(
            config.FEATURES_IMPORTANCE_OUTPUT_classification if task == 'classification' 
            else config.FEATURES_IMPORTANCE_OUTPUT_regression,
            f"{model}_{score_type.lower()}_feature_importances.xlsx"
        )
        if os.path.exists(filepath):
            df = pd.read_excel(filepath)
            importances[model] = df
    
    return importances

def aggregate_feature_importance(importances):
    """
    Aggregate feature importances from different models
    """
    # Create dictionaries to store aggregated importance and std for each feature
    feature_scores = {}
    feature_stds = {}
    
    # Process each model's importance scores
    for imp_df in importances.values():
        # Normalize importance scores for this model
        total_importance = imp_df['Importance'].sum()
        normalized_scores = imp_df['Importance'] / total_importance
        normalized_stds = imp_df['Importance_STD'] / total_importance
        
        # Add to feature scores
        for feature, score, std in zip(imp_df['Feature'], normalized_scores, normalized_stds):
            if feature not in feature_scores:
                feature_scores[feature] = []
                feature_stds[feature] = []
            feature_scores[feature].append(score)
            feature_stds[feature].append(std)
    
    # Calculate mean importance and propagate uncertainties for each feature
    aggregated_scores = {}
    aggregated_stds = {}
    for feature in feature_scores:
        # Mean importance across models
        aggregated_scores[feature] = np.mean(feature_scores[feature])
        # Combined standard deviation (propagation of uncertainty)
        model_std = np.mean(feature_stds[feature])  # Mean of per-model std
        between_model_std = np.std(feature_scores[feature])  # Std between models
        aggregated_stds[feature] = np.sqrt(model_std**2 + between_model_std**2)
    
    # Convert to DataFrame and sort
    agg_df = pd.DataFrame({
        'Feature': list(aggregated_scores.keys()),
        'Mean_Importance': list(aggregated_scores.values()),
        'Std_Importance': list(aggregated_stds.values())
    }).sort_values('Mean_Importance', ascending=False)
    
    return agg_df

def select_top_features(agg_importance, threshold_percentile=20):
    """
    Select top features based on importance scores
    threshold_percentile: Keep features with importance in top X percentile
    """
    threshold = np.percentile(agg_importance['Mean_Importance'], 100 - threshold_percentile)
    selected_features = agg_importance[agg_importance['Mean_Importance'] >= threshold]['Feature'].tolist()
    return selected_features

def save_selected_features(selected_features, score_type, task='classification'):
    """Save selected features to a file"""
    output_dir = (config.FEATURES_IMPORTANCE_OUTPUT_classification 
                 if task == 'classification' 
                 else config.FEATURES_IMPORTANCE_OUTPUT_regression)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"selected_features_{score_type.lower()}.txt")
    with open(output_file, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Selected features saved to: {output_file}")

def plot_feature_importance_comparison(importances, score_type, task='classification', n_features=10):
    """
    Plot feature importances comparison across models.
    
    Parameters:
    -----------
    importances : dict
        Dictionary containing feature importances from different models
    score_type : str
        Type of score being predicted (FRIED or FRAGIRE18)
    task : str
        Type of task (classification or regression)
    n_features : int, optional (default=10)
        Number of top features to plot
    """
    # Get mean importance across models for each feature
    mean_importance = {}
    for feature in importances['lightgbm']['Feature'].unique():
        feature_importances = []
        for model in importances.keys():
            importance = importances[model][importances[model]['Feature'] == feature]['Importance'].values[0]
            feature_importances.append(importance)
        mean_importance[feature] = np.mean(feature_importances)
    
    # Sort features by mean importance
    sorted_features = sorted(mean_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:n_features]
    
    # Prepare data for plotting
    features = [x[0] for x in top_features]
    model_importances = {model: [] for model in importances.keys()}
    for feature in features:
        for model in importances.keys():
            importance = importances[model][importances[model]['Feature'] == feature]['Importance'].values[0]
            model_importances[model].append(importance)
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    r1 = np.arange(len(features))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    plt.bar(r1, model_importances['lightgbm'], width=bar_width, label='LightGBM', color='lightgreen')
    plt.bar(r2, model_importances['xgboost'], width=bar_width, label='XGBoost', color='blue')
    plt.bar(r3, model_importances['catboost'], width=bar_width, label='CatBoost', color='red')
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Top {n_features} Feature Importances Comparison - {score_type}')
    plt.xticks([r + bar_width for r in range(len(features))], features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = (config.VISUALIZATION_OUTPUT_Classification 
                 if task == 'classification' 
                 else config.VISUALIZATION_OUTPUT_Regression)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plot
    plt.savefig(
        os.path.join(output_dir, f'feature_importance_comparison_{score_type.lower()}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

def main(score_type, task='classification', threshold_percentile=20):
    """
    Main function to perform feature selection
    """
    # Load data
    X, y, feature_names = data_loader.load_data(score_type, task)
    
    # Initialize models based on task
    if task == 'classification':
        models = {
            'lightgbm': LGBMClassifier(random_state=42),
            'xgboost': XGBClassifier(random_state=42),
            'catboost': CatBoostClassifier(random_state=42)
        }
    else:  # regression
        models = {
            'lightgbm': LGBMRegressor(random_state=42),
            'xgboost': XGBRegressor(random_state=42),
            'catboost': CatBoostRegressor(random_state=42)
        }
    
    # Calculate and save feature importances for each model
    for name, model in models.items():
        model.fit(X, y)
        importance = model.feature_importances_
        importance_std = np.zeros_like(importance)  # No std for single run
        
        # Save feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Importance_STD': importance_std
        }).sort_values('Importance', ascending=False)
        
        output_dir = (config.FEATURES_IMPORTANCE_OUTPUT_classification 
                     if task == 'classification' 
                     else config.FEATURES_IMPORTANCE_OUTPUT_regression)
        os.makedirs(output_dir, exist_ok=True)
        importance_df.to_excel(
            os.path.join(output_dir, f"{name}_{score_type.lower()}_feature_importances.xlsx"),
            index=False
        )
    
    # Load feature importances
    importances = load_feature_importances(score_type, task)
    
    # Aggregate importances and select top features
    agg_importance = aggregate_feature_importance(importances)
    selected_features = select_top_features(agg_importance, threshold_percentile)
    
    # Save selected features
    save_selected_features(selected_features, score_type, task)
    
    # Plot feature importance comparison
    plot_feature_importance_comparison(importances, score_type, task)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Feature selection based on model importances')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      help='Type of score to predict (FRIED or FRAGIRE18)')
    parser.add_argument('--task', type=str, default='classification',
                      help='Type of task (classification or regression)')
    parser.add_argument('--threshold_percentile', type=float, default=20,
                      help='Percentile threshold for feature selection')
    args = parser.parse_args()
    
    main(args.score_type, args.task, args.threshold_percentile)
