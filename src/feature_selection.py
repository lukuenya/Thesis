import pandas as pd
import numpy as np
import joblib
import os
import data_loader
import config
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_importances(score_type, task='classification'):
    """
    Load feature importance files for all models
    """
    importances = {}
    models = ['lightgbm', 'xgboost', 'catboost', 'randomforest']
    
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
            'catboost': CatBoostClassifier(random_state=42),
            'randomforest': RandomForestClassifier(random_state=42, n_jobs=-1)
        }
    else:  # regression
        models = {
            'lightgbm': LGBMRegressor(random_state=42),
            'xgboost': XGBRegressor(random_state=42),
            'catboost': CatBoostRegressor(random_state=42),
            'randomforest': RandomForestRegressor(random_state=42, n_jobs=-1)
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Feature selection based on model importances')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18', 'CHUTE_6M', 'CHUTE_12M'],
                      help='Score type to use for feature selection')
    parser.add_argument('--task', type=str, default='classification',
                      choices=['classification', 'regression'],
                      help='Task type (classification or regression)')
    parser.add_argument('--threshold_percentile', type=float, default=20,
                      help='Percentile threshold for feature selection')
    args = parser.parse_args()
    
    main(args.score_type, args.task, args.threshold_percentile)
