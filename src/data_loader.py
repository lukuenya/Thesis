# src/data_loader.py

import pandas as pd
import numpy as np
import config


def load_data(target_score='FRIED'):
    """
    Load the dataset and prepare X, y for model training
    
    Parameters:
    -----------
    target_score : str, optional (default='FRIED')
        Which score to predict: 'FRIED' or 'FRAGIRE18'
    
    Returns:
    --------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable (FRIED or FRAGIRE18 score)
    """
    df = pd.read_excel(config.TRAINING_FILE)
    
    if target_score.upper() == 'FRIED':
        # Drop FRIED-related columns and FRAGIRE18 score
        X = df.drop(config.COLS_TO_DROP_FRIED_SCORE, axis=1)
        y = df.Fried_Score_FRIED_TOTAL_Version_1
    else:  # FRAGIRE18
        # Drop FRAGIRE18 score and FRIED-related columns
        X = df.drop(config.COLS_TO_DROP_FRAILTY_SCORE, axis=1)
        y = df.Frailty_Score_FRAGIRE18_SQ001

    # Remove any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    return X, y

    #     # read the training data with folds
    # df = pd.read_csv(config.TRAINING_FILE)

    # # Training data is where kfold is not equal to provided fold. We reset the index.
    # df_train = df[df.kfold != fold].reset_index(drop=True)

    # # Validation data is where kfold is equal to provided fold.
    # df_valid = df[df.kfold == fold].reset_index(drop=True)

    # # drop some columns
    # x_train = df_train.drop(config.COLS_TO_DROP, axis=1).values

    # # Target is 'Frailty_State' column
    # y_train = df_train.Frailty_State.values

    # # For Validation we have
    # x_valid = df_valid.drop(config.COLS_TO_DROP, axis=1).values

    # y_valid = df_valid.Frailty_State.values

    # return x_train, y_train, x_valid, y_valid
