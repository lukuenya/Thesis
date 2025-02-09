import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold
import data_loader
import config

# Default parameters for each model and score type
MODEL_PARAMS = {
    'FRIED': {
        'classification': {
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
        'regression': {
            'lightgbm': {
                'reg_alpha': 1.6266623378395724e-05,
                'reg_lambda': 2.7803224284458052e-08,
                'n_estimators': 209,
                'num_leaves': 349,
                'max_depth': 10,
                'learning_rate': 0.007303587470843107,
                'boosting_type': 'dart'
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
        }
    },
    'FRAGIRE18': {
        'classification': {
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
        'regression': {
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
        }
    }
}

def get_feature_importances(score_type, model_name, task='classification'):
    """
    Get feature importances from a model using cross-validation
    
    Parameters:
    -----------
    score_type : str
        Type of score to predict (FRIED or FRAGIRE18)
    model_name : str
        Name of the model to use (lightgbm, xgboost, or catboost)
    task : str
        Type of task (classification or regression)
    """
    # Load data
    X, y, feature_names = data_loader.load_data(score_type, task)
    
    # Initialize model with parameters
    if task == 'classification':
        if model_name == 'lightgbm':
            model = LGBMClassifier(**MODEL_PARAMS[score_type][task]['lightgbm'])
        elif model_name == 'xgboost':
            model = XGBClassifier(**MODEL_PARAMS[score_type][task]['xgboost'])
        else:  # catboost
            model = CatBoostClassifier(**MODEL_PARAMS[score_type][task]['catboost'])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:  # regression
        if model_name == 'lightgbm':
            model = LGBMRegressor(**MODEL_PARAMS[score_type][task]['lightgbm'])
        elif model_name == 'xgboost':
            model = XGBRegressor(**MODEL_PARAMS[score_type][task]['xgboost'])
        else:  # catboost
            model = CatBoostRegressor(**MODEL_PARAMS[score_type][task]['catboost'])
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Get feature importances using cross-validation
    importances = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)])
        
        importances.append(model.feature_importances_)
    
    # Calculate mean and std of feature importances
    importances = np.array(importances)
    importance_mean = importances.mean(axis=0)
    importance_std = importances.std(axis=0)
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_mean,
        'Importance_STD': importance_std
    }).sort_values('Importance', ascending=False)
    
    # Save feature importances
    output_dir = (config.FEATURES_IMPORTANCE_OUTPUT_classification 
                 if task == 'classification' 
                 else config.FEATURES_IMPORTANCE_OUTPUT_regression)
    os.makedirs(output_dir, exist_ok=True)
    importance_df.to_excel(
        os.path.join(output_dir, f"{model_name}_{score_type.lower()}_feature_importances.xlsx"),
        index=False
    )
    
    return importance_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Get feature importances from models')
    parser.add_argument('--score_type', type=str, default='FRIED',
                      help='Type of score to predict (FRIED or FRAGIRE18)')
    parser.add_argument('--model_name', type=str, default='lightgbm',
                      choices=['lightgbm', 'xgboost', 'catboost'],
                      help='Model to use for feature importance calculation')
    parser.add_argument('--task', type=str, default='classification',
                      choices=['classification', 'regression'],
                      help='Type of task (classification or regression)')
    args = parser.parse_args()
    
    get_feature_importances(args.score_type, args.model_name, args.task)
