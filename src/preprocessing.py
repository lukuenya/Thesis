# src/preprocessing.py

import pandas as pd 
import numpy as np

from sklearn.feature_selection import VarianceThreshold


def drop_high_missing(df, threshold):
    """
    Drop columns with more than threshold missing values
    """
    frac_missing = df.isna().mean() # fraction of NaNs/column
    cols_to_drop = frac_missing[frac_missing > threshold].index
    return df.drop(columns=cols_to_drop)


def drop_near_zero_variance(df, freq_threshold=0.95):
    """
    Remove quasi-constant features where the most frequent value 
    appears more than freq_threshold of the time
    """
    quasi_constant_mask = df.apply(lambda x: x.value_counts(normalize=True).iloc[0] < freq_threshold)
    return df.loc[:, quasi_constant_mask]


def drop_correlated_features(
    df, 
    method='pearson',
    correlation_threshold=0.95):

    corr_matrix = df.corr(method=method).abs()

    # Select upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]

    return df.drop(columns=to_drop)



