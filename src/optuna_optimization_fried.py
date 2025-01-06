# src/optuna_optimization_reg.py

import data_loader
import config

import numpy as np
from sklearn import metrics, model_selection
from lightgbm import LGBMRegressor
import optuna
from functools import partial
from math import sqrt
import joblib


def objective(trial, x, y):
    # Define hyperparameters for each model
    n_estimators = trial.suggest_int('n_estimators', 30, 300)
    num_leaves = trial.suggest_int('num_leaves', 100, 500)
    max_depth = trial.suggest_int('max_depth', 8, 12)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
    boosting_type = trial.suggest_categorical(
        'boosting_type', ['dart', 'gbdt'])

    model = LGBMRegressor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        boosting_type=boosting_type)
    
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

    rmse = []

    for idx in kf.split(X=x, y=y):
        train_idx, valid_idx = idx[0], idx[1]
        x_train, y_train = x[train_idx], y[train_idx]
        x_valid, y_valid = x[valid_idx], y[valid_idx]

        model.fit(x_train, y_train)

        preds = model.predict(x_valid)

        fold_rmse = sqrt(metrics.mean_squared_error(y_valid, preds))

        rmse.append(fold_rmse)

    mean_rmse = np.mean(rmse)
    std_rmse = np.std(rmse)

    # Store the std as a trial attribute
    trial.set_user_attr('std_rmse', std_rmse)

    return mean_rmse


if __name__ == "__main__":
    # Load data
    X, y = data_loader.load_data()

    optimization_function = partial(
        objective,
        x=X,
        y=y
    )

    study = optuna.create_study(direction='minimize')
    study.optimize(optimization_function, n_trials=25, show_progress_bar=True)

    # Print the best parameters
    print(f"Best parameters: {study.best_params}")

    # Print the best score
    print(f"Best score: {study.best_value:.2f}")

    # Get the standard deviation of the best trial
    best_trial = study.best_trial
    std_rmse = best_trial.user_attrs['std_rmse']
    print(f"Standard deviation: {std_rmse:.2f}")

    # Retrain the model with the best parameters on the entire dataset
    best_params = study.best_params
    best_model = LGBMRegressor(**best_params)
    best_model.fit(X, y)

    # Save the best model
    joblib.dump(best_model, config.MODEL_OUTPUT + 'regressor_model_fried.joblib')

