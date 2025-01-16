import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
            'boosting_type': 'gbdt'
        },
        'xgboost': {
            'n_estimators': 151,
            'alpha': 1.2895984520282453,
            'max_depth': 7,
            'eta': 0.22888318411754455,
            'min_child_weight': 0,
            'subsample': 0.7547159187812194,
            'colsample_bytree': 0.8713219606921598
        },
        'catboost': {
            'iterations': 297,
            'depth': 4,
            'learning_rate': 0.1126205922460068,
            'l2_leaf_reg': 2.408872205466443,
            'bagging_temperature': 0.9718611384971728,
            'random_strength': 1.6404371699627177e-08
        }
    },
    'FRAGIRE18': {
        'lightgbm': {
            'reg_alpha': 5.504139817600005,
            'reg_lambda': 1.2579874764708709e-08,
            'n_estimators': 265,
            'num_leaves': 364,
            'max_depth': 7,
            'learning_rate': 0.09870445582078947,
            'boosting_type': 'dart'
        },
        'xgboost': {
            'n_estimators': 186,
            'alpha': 0.5669722800468638,
            'max_depth': 8,
            'eta': 0.11499078868079486,
            'min_child_weight': 2,
            'subsample': 0.9999191623214145,
            'colsample_bytree': 0.9969053373589476
        },
        'catboost': {
            'iterations': 179,
            'depth': 12,
            'learning_rate': 0.21823430934181487,
            'l2_leaf_reg': 0.27821259087472894,
            'bagging_temperature': 0.7744608687893274,
            'random_strength': 0.026497827883302558
        }
    }
}

def get_feature_importances(score_type, model_name):
    """Get feature importances from a model"""
    # Load data
    X, y, feature_names = data_loader.load_data(score_type)
    
    # Get model parameters for this score type
    params = MODEL_PARAMS[score_type][model_name]
    
    # Create and train model
    if model_name == "lightgbm":
        model = LGBMClassifier(**params)
    elif model_name == "xgboost":
        model = XGBClassifier(**params)
    elif model_name == "catboost":
        model = CatBoostClassifier(**params)
    
    # Train model
    model.fit(X, y)
    
    # Get and save feature importances
    feature_importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    
    # Save to file
    importance_filename = f"{model_name}_{score_type.lower()}_feature_importances.xlsx"
    os.makedirs(config.FEATURES_IMPORTANCE_OUTPUT, exist_ok=True)
    feature_importances_df.to_excel(
        os.path.join(config.FEATURES_IMPORTANCE_OUTPUT, importance_filename),
        index=False
    )
    print(f"Feature importances saved to {importance_filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Get feature importances from models')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      choices=['FRIED', 'FRAGIRE18'],
                      help='Which score type to analyze')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost'],
                      help='Which model to use')
    
    args = parser.parse_args()
    get_feature_importances(args.score_type, args.model_name)
