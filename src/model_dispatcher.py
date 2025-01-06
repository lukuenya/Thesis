# model_dispatcher.py

from lightgbm import LGBMRegressor

# Dictionary of models for training
MODELS = {
    "lightgbm": LGBMRegressor(
        objective="regression",
        metric="rmse",
        verbosity=-1,
        boosting_type="gbdt",
        n_estimators=100
    )
}
