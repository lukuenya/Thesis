import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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
            'boosting_type': 'dart',
            'is_unbalance': True
        },
        'xgboost': {
            'n_estimators': 186,
            'alpha': 0.5669722800468638,
            'max_depth': 8,
            'eta': 0.11499078868079486,
            'min_child_weight': 2,
            'subsample': 0.9999191623214145,
            'colsample_bytree': 0.9969053373589476,
            'scale_pos_weight': 4.59375
        },
        'catboost': {
            'iterations': 179,
            'depth': 12,
            'learning_rate': 0.21823430934181487,
            'l2_leaf_reg': 0.27821259087472894,
            'bagging_temperature': 0.7744608687893274,
            'random_strength': 0.026497827883302558,
            'auto_class_weights': 'Balanced'
        }
    }
}

def get_feature_importances(score_type, model_name):
    """Get feature importances from a model using StratifiedKFold"""
    # Load data
    X, y, feature_names = data_loader.load_data(score_type)
    
    # Get model parameters for this score type
    params = MODEL_PARAMS[score_type][model_name]
    
    # Initialize StratifiedKFold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store feature importances from each fold
    feature_importances_folds = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        
        # Create and train model
        if model_name == "lightgbm":
            model = LGBMClassifier(**params)
        elif model_name == "xgboost":
            model = XGBClassifier(**params)
        elif model_name == "catboost":
            model = CatBoostClassifier(**params)
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        
        # Get feature importances for this fold
        feature_importances_folds.append(model.feature_importances_)
    
    # Calculate mean and std of feature importances across folds
    feature_importances_mean = np.mean(feature_importances_folds, axis=0)
    feature_importances_std = np.std(feature_importances_folds, axis=0)
    
    # Create DataFrame with feature importances and their standard deviations
    feature_importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances_mean,
        'Importance_STD': feature_importances_std
    }).sort_values('Importance', ascending=False)
    
    # Save to file
    importance_filename = f"{model_name}_{score_type.lower()}_feature_importances.xlsx"
    os.makedirs(config.FEATURES_IMPORTANCE_OUTPUT, exist_ok=True)
    feature_importances_df.to_excel(
        os.path.join(config.FEATURES_IMPORTANCE_OUTPUT, importance_filename),
        index=False
    )
    print(f"Feature importances saved to {importance_filename}")
    
    # Print cross-validation information
    print(f"\nFeature importance statistics from {n_splits}-fold cross-validation:")
    for i in range(min(10, len(feature_names))):  # Print top 10 features
        feat = feature_importances_df.iloc[i]
        print(f"{feat['Feature']}: {feat['Importance']:.4f} ± {feat['Importance_STD']:.4f}")

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
