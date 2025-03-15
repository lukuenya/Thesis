import pandas as pd
import numpy as np
import joblib
import os
import data_loader
import config
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_importances(score_type):
    """
    Load feature importance files for all models
    """
    importances = {}
    models = ['lightgbm', 'xgboost', 'catboost', 'randomforest']
    
    for model in models:
        filepath = os.path.join(
            config.FEATURE_IMPORTANCE_DIR,
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
    # Initialize dictionary to store normalized scores
    normalized_importances = {}
    scaler = MinMaxScaler()
    
    for model_name, imp_df in importances.items():
        # Reshape for scaler and normalize importance scores
        importance_scores = imp_df['Importance'].values.reshape(-1, 1)
        normalized_scores = scaler.fit_transform(importance_scores).ravel()
        
        # Scale standard deviations using the same scaling factor
        # This preserves the relative uncertainty while keeping the scale consistent
        scale_factor = (scaler.data_max_ - scaler.data_min_)[0]
        normalized_stds = imp_df['Importance_STD'].values / scale_factor if scale_factor != 0 else imp_df['Importance_STD'].values
        
        normalized_importances[model_name] = pd.DataFrame({
            'Feature': imp_df['Feature'],
            'Normalized_Importance': normalized_scores,
            'Normalized_STD': normalized_stds
        })
    
    # Calculate mean and std across models for each feature
    all_features = normalized_importances[list(normalized_importances.keys())[0]]['Feature']
    mean_importance = pd.DataFrame({
        'Feature': all_features,
        'Mean_Importance': 0.0,
        'Std_Importance': 0.0
    })
    
    for feature in all_features:
        feature_scores = []
        for model_scores in normalized_importances.values():
            score = model_scores[model_scores['Feature'] == feature]['Normalized_Importance'].values[0]
            feature_scores.append(score)
        
        mean_importance.loc[mean_importance['Feature'] == feature, 'Mean_Importance'] = np.mean(feature_scores)
        mean_importance.loc[mean_importance['Feature'] == feature, 'Std_Importance'] = np.std(feature_scores)
    
    return mean_importance.sort_values('Mean_Importance', ascending=False)

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
    filepath = os.path.join(
        config.FEATURE_IMPORTANCE_DIR,
        f"selected_features_{score_type.lower()}_classification.txt"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print(f"Selected features saved to: {filepath}")

def main(score_type, threshold_percentile=20):
    """
    Main function to perform feature selection
    """
    # Load feature importances for all models
    importances = load_feature_importances(score_type)
    
    if not importances:
        print(f"No feature importance files found for {score_type}")
        return
    
    # Aggregate importance scores across models
    agg_importance = aggregate_feature_importance(importances)
    
    # Select top features
    selected_features = select_top_features(agg_importance, threshold_percentile)
    
    # Save selected features
    save_selected_features(selected_features, score_type)
    
    # Plot feature importances
    plt.figure(figsize=(6, 8), dpi=300)  # High resolution
    
    # Sort features by mean importance
    sorted_agg = agg_importance.sort_values(by='Mean_Importance', ascending=False).reset_index(drop=True)
    
    # Plot top 30 features (or all if less than 30)
    n_features = min(30, len(sorted_agg))
    subset = sorted_agg.iloc[:n_features]
    
    # Plot with error bars
    plt.errorbar(
        x=range(n_features),
        y=subset['Mean_Importance'],
        yerr=subset['Std_Importance'],
        fmt='o',
        capsize=4,
        ecolor='red',
        markersize=8,
        markerfacecolor='blue',
        markeredgecolor='black'
    )
    
    plt.xticks(range(n_features), subset['Feature'], rotation=90)
    plt.ylabel('Normalized Importance')
    plt.tight_layout()
    
    # Save plot
    output_dir = config.FEATURE_IMPORTANCE_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"feature_importance_{score_type.lower()}.png"))
    plt.close()
    
    return selected_features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Feature selection based on model importances')
    parser.add_argument('--score_type', type=str, default='FRIED',
                        help='Score to predict: FRIED, FRAGIRE18, CHUTE_6M, or CHUTE_12M')
    parser.add_argument('--threshold', type=int, default=20,
                        help='Percentile threshold for feature selection (1-100)')
    
    args = parser.parse_args()
    main(args.score_type, args.threshold)
