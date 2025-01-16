# model_dispatcher.py

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Class weight ratios calculated from data:
# Fried = 4.59375 (115/32)
# FRAGIRE18 = 3.585365 (103/40)

# Define base models with classification-specific parameters
MODELS = {
    "lightgbm": lgb.LGBMClassifier(
        is_unbalance=True,  # Automatically handles imbalanced datasets
        objective='binary',
        metric='auc',
        n_jobs=-1,
        random_state=42,
        verbose=-1
    ),
    
    "xgboost": xgb.XGBClassifier(
        booster='dart',
        objective='binary:logistic',
        scale_pos_weight=4.59375,  # Using Fried ratio as it's more imbalanced
        max_delta_step=1,  # Helps with class imbalance
        eval_metric='auc',
        n_jobs=-1,
        random_state=42,
        verbosity=0
    ),
    
    "catboost": CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
    #  # [majority_class_weight, minority_class_weight]
        random_seed=42,
        verbose=False
    )
}
