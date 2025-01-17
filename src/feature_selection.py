import pandas as pd
import numpy as np
import joblib
import os
import data_loader
import config
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_importances(score_type):
    """
    Load feature importance files for all models
    """
    importances = {}
    models = ['lightgbm', 'xgboost', 'catboost']
    
    for model in models:
        filepath = os.path.join(config.FEATURES_IMPORTANCE_OUTPUT, 
                              f"{model}_{score_type.lower()}_feature_importances.xlsx")
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

def save_selected_features(selected_features, score_type):
    """Save selected features to a file"""
    os.makedirs(config.FEATURES_IMPORTANCE_OUTPUT, exist_ok=True)
    output_file = os.path.join(config.FEATURES_IMPORTANCE_OUTPUT, f"selected_features_{score_type.lower()}.txt")
    with open(output_file, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"Selected features saved to: {output_file}")

def plot_feature_importance_comparison(importances, score_type, n_features=10):
    """
    Plot feature importances comparison across models.
    
    Parameters:
    -----------
    importances : dict
        Dictionary containing feature importance DataFrames for each model
    score_type : str
        Type of score (FRIED or FRAGIRE18)
    n_features : int
        Number of top features to plot
    """
    # Get aggregated importances to determine top features
    agg_df = aggregate_feature_importance(importances)
    top_features = agg_df['Feature'].head(n_features).tolist()
    
    # Prepare data for plotting
    plot_data = []
    n_folds = 5  # Number of CV folds
    
    for model_name, imp_df in importances.items():
        # Normalize importances
        total_importance = imp_df['Importance'].sum()
        imp_df = imp_df.copy()
        imp_df['Normalized_Importance'] = imp_df['Importance'] / total_importance
        # Convert to Standard Error
        imp_df['Error'] = (imp_df['Importance_STD'] / total_importance) / np.sqrt(n_folds)
        
        # Filter for top features and add model name
        model_data = imp_df[imp_df['Feature'].isin(top_features)].copy()
        model_data['Model'] = model_name.upper()
        plot_data.append(model_data)
    
    # Combine all data
    plot_df = pd.concat(plot_data, ignore_index=True)
    
    # Create the plot
    plt.figure(figsize=(12, 6))  # Reduced height
    
    # Create grouped bar plot
    bar_width = 0.25
    models = plot_df['Model'].unique()
    x = np.arange(len(top_features))
    
    colors = ['#2196F3', '#FFA726', '#66BB6A']  # More muted colors
    
    for i, (model, color) in enumerate(zip(models, colors)):
        model_data = plot_df[plot_df['Model'] == model]
        # Sort features in same order as top_features
        model_data = model_data.set_index('Feature').loc[top_features].reset_index()
        
        plt.bar(x + i*bar_width, 
               model_data['Normalized_Importance'],
               bar_width,
               label=model,
               color=color,
               alpha=0.8)  
        
        # Add error bars with smaller caps
        plt.errorbar(x + i*bar_width, 
                    model_data['Normalized_Importance'],
                    yerr=model_data['Error'],
                    fmt='none',
                    color='black',
                    capsize=3,
                    capthick=1,
                    linewidth=1,
                    alpha=0.5)  
    
    plt.xlabel('Features', fontsize=10)
    plt.ylabel('Normalized Importance', fontsize=10)
    plt.title(f'Feature Importance - {score_type}', fontsize=12, pad=20)
    
    # Adjust x-axis labels
    plt.xticks(x + bar_width, top_features, rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    # Add legend with smaller font
    plt.legend(fontsize=9, loc='upper right')
    
    # Adjust grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Tight layout with more space for feature names
    plt.tight_layout()
    
    # Save plot
    os.makedirs(config.VISUALIZATION_OUTPUT, exist_ok=True)
    plt.savefig(
        os.path.join(config.VISUALIZATION_OUTPUT, f'feature_importance_comparison_{score_type.lower()}.svg'),
        format='svg',
        bbox_inches='tight',
        dpi=300
    )
    plt.close()

def main(score_type, threshold_percentile=20):
    """
    Main function to perform feature selection
    """
    print(f"Loading feature importances for {score_type}...")
    importances = load_feature_importances(score_type)
    
    if not importances:
        print("No feature importance files found. Please run 'get_feature_importances' first.")
        return
    
    print("Aggregating feature importances across models...")
    agg_importance = aggregate_feature_importance(importances)
    
    print(f"\nTop 10 most important features:")
    print(agg_importance.head(10))
    
    print(f"\nCreating feature importance comparison plot...")
    plot_feature_importance_comparison(importances, score_type)
    
    print(f"\nSelecting top {threshold_percentile}% features...")
    selected_features = select_top_features(agg_importance, threshold_percentile)
    
    print(f"Selected {len(selected_features)} features out of {len(agg_importance)}")
    
    # Save selected features
    save_selected_features(selected_features, score_type)
    
    return selected_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Feature selection based on model importances')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18'],
                      help='Which score to use for feature selection')
    parser.add_argument('--threshold_percentile', type=int, default=20,
                      help='Keep features with importance in top X percentile')
    
    args = parser.parse_args()
    selected_features = main(args.score_type, args.threshold_percentile)
