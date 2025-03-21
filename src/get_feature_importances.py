import os
import pandas as pd
import numpy as np
import argparse
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import data_loader
import config

# Default parameters for each model and score type
MODEL_PARAMS = {
    'FRIED': {
        'lightgbm': {
            'reg_alpha': 1.6266623378395724e-05,
            'reg_lambda': 2.7803224284458052e-08,
            'n_estimators': 209,
            'num_leaves': 349,
            'max_depth': 10,
            'learning_rate': 0.007303587470843107,
            'boosting_type': 'dart',
            'is_unbalance': True
        },
        'xgboost': {
            'n_estimators': 151,
            'alpha': 1.2895984520282453,
            'max_depth': 7,
            'eta': 0.22888318411754455,
            'min_child_weight': 0,
            'subsample': 0.7547159187812194,
            'colsample_bytree': 0.8713219606921598,
            'scale_pos_weight': 4.59375
        },
        'catboost': {
            'iterations': 297,
            'depth': 4,
            'learning_rate': 0.1126205922460068,
            'l2_leaf_reg': 2.408872205466443,
            'bagging_temperature': 0.9718611384971728,
            'random_strength': 1.6404371699627177e-08,
            'auto_class_weights': 'Balanced'
        },
        'randomforest': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced'
        }
    },
    'FRAGIRE18': {
        'lightgbm': {
            'reg_alpha': 1.6266623378395724e-05,
            'reg_lambda': 2.7803224284458052e-08,
            'n_estimators': 209,
            'num_leaves': 349,
            'max_depth': 10,
            'learning_rate': 0.007303587470843107,
            'boosting_type': 'dart',
            'is_unbalance': True
        },
        'xgboost': {
            'n_estimators': 151,
            'alpha': 1.2895984520282453,
            'max_depth': 7,
            'eta': 0.22888318411754455,
            'min_child_weight': 0,
            'subsample': 0.7547159187812194,
            'colsample_bytree': 0.8713219606921598,
            'scale_pos_weight': 4.59375
        },
        'catboost': {
            'iterations': 297,
            'depth': 4,
            'learning_rate': 0.1126205922460068,
            'l2_leaf_reg': 2.408872205466443,
            'bagging_temperature': 0.9718611384971728,
            'random_strength': 1.6404371699627177e-08,
            'auto_class_weights': 'Balanced'
        },
        'randomforest': {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced'
        }
    }
}

def get_feature_importances(score_type, model_name, imputation=True):
    """
    Get feature importances from a model using cross-validation
    
    Parameters:
    -----------
    score_type : str
        Type of score to predict (FRIED or FRAGIRE18)
    model_name : str
        Name of the model to use (lightgbm, xgboost, catboost, or randomforest)
    imputation : bool
        Whether to use imputed data or raw data
    """
    # Load data
    X, y = data_loader.load_data(score_type, imputation=imputation)
    
    # Note: NaN values are already handled in data_loader.py
    
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    if isinstance(y, pd.Series):
        y = y.values
        
    # Initialize model
    if model_name == 'lightgbm':
        model = LGBMClassifier(**MODEL_PARAMS[score_type][model_name], random_state=42)
    elif model_name == 'xgboost':
        model = XGBClassifier(**MODEL_PARAMS[score_type][model_name], random_state=42)
    elif model_name == 'catboost':
        model = CatBoostClassifier(**MODEL_PARAMS[score_type][model_name], random_state=42, verbose=0)
    elif model_name == 'randomforest':
        model = RandomForestClassifier(**MODEL_PARAMS[score_type][model_name], random_state=42, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Cross-validation (5-fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    importances = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get feature importances
        if model_name == 'lightgbm':
            imp = model.feature_importances_
        elif model_name == 'xgboost':
            imp = model.feature_importances_
        elif model_name == 'catboost':
            imp = model.feature_importances_
        elif model_name == 'randomforest':
            imp = model.feature_importances_
        
        importances.append(imp)
    
    # Average feature importances
    mean_importances = np.mean(importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Get output paths based on imputation
    output_paths = config.get_output_paths(imputation=imputation)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_paths["feature_importances"], exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(
        output_paths["feature_importances"],
        f"importance_{model_name}_{score_type.lower()}_classification.csv"
    )
    importance_df.to_csv(output_file, index=False)
    print(f"Feature importances saved to {output_file}")
    
    # Save top features to a text file
    top_features = importance_df['feature'].head(100).tolist()
    top_features_file = os.path.join(
        output_paths["feature_importances"],
        f"selected_features_{score_type.lower()}_classification.txt"
    )
    with open(top_features_file, 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    
    print(f"Top 100 features saved to {top_features_file}")
    return importance_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get feature importances from models')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18'],
                      help='Score type to predict')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost', 'randomforest'],
                      help='Model to use')
    parser.add_argument('--no_imputation', action='store_true',
                      help='Use raw data without imputation')
    args = parser.parse_args()
    
    get_feature_importances(args.score_type, args.model_name, imputation=not args.no_imputation)
