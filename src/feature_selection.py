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
    # Create a dictionary to store aggregated importance for each feature
    feature_scores = {}
    
    print("\nNormalization process:")
    # Process each model's importance scores
    for model_name, imp_df in importances.items():
        print(f"\n{model_name.upper()}:")
        print(f"Sum of original importances: {imp_df['Importance'].sum():.4f}")
        
        # Normalize importance scores for this model
        normalized_scores = imp_df['Importance'] / imp_df['Importance'].sum()
        
        # Print top 3 features before and after normalization
        print("\nTop 3 features:")
        for i in range(min(3, len(imp_df))):
            orig_score = imp_df['Importance'].iloc[i]
            norm_score = normalized_scores.iloc[i]
            feat_name = imp_df['Feature'].iloc[i]
            print(f"{feat_name}:")
            print(f"  Original: {orig_score:.4f}")
            print(f"  Normalized: {norm_score:.4f} ({norm_score*100:.2f}%)")
        
        # Add to feature scores
        for feature, score in zip(imp_df['Feature'], normalized_scores):
            if feature not in feature_scores:
                feature_scores[feature] = []
            feature_scores[feature].append(score)
    
    # Calculate mean importance for each feature
    aggregated_scores = {
        feature: np.mean(scores) 
        for feature, scores in feature_scores.items()
    }
    
    # Convert to DataFrame and sort
    agg_df = pd.DataFrame({
        'Feature': list(aggregated_scores.keys()),
        'Mean_Importance': list(aggregated_scores.values())
    }).sort_values('Mean_Importance', ascending=False)
    
    # Print top 3 features after aggregation
    print("\nTop 3 features after aggregating normalized scores:")
    for i in range(min(3, len(agg_df))):
        feat = agg_df['Feature'].iloc[i]
        score = agg_df['Mean_Importance'].iloc[i]
        print(f"{feat}: {score:.4f} ({score*100:.2f}%)")
    
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
