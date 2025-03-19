# src/preprocessing.py

import pandas as pd 
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


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


def cvbknn_imputer(X, n_splits=5, k_values=[3, 5, 7,10], verbose=False):
    """
    Cross-validated KNN imputation to find optimal k
    """
    # Convert to numpy array and ensure float type
    X = np.asarray(X, dtype=np.float64)
    
    best_k = None
    min_error = float('inf')
    
    # Get mask of missing values
    mask_nan = np.isnan(X)
    n_missing = np.sum(mask_nan)
    
    if n_missing == 0:
        return X, k_values[0]
    
    if verbose:
        print(f"Missing values: {n_missing}")
    
    # Cross-validation to find optimal k
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for k in k_values:
        cv_errors = []
        
        for train_idx, test_idx in kf.split(X):
            X_train = X[train_idx]
            X_test = X[test_idx].copy()  # Make a copy to preserve original values
            
            # Create artificial missing values in test set
            test_mask = ~np.isnan(X_test)  # Only consider non-missing values
            artificial_mask = np.random.random(X_test.shape) < 0.2  # 20% artificial missing
            artificial_mask = artificial_mask & test_mask  # Only mask non-missing values
            
            # Store original values and create artificial NaNs
            X_test_original = X_test.copy()
            X_test[artificial_mask] = np.nan
            
            # Fit imputer on training data
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(X_train)
            
            # Impute test data
            X_test_imputed = imputer.transform(X_test)
            
            # Calculate error only on artificially masked values
            if np.sum(artificial_mask) > 0:
                error = np.sqrt(np.mean((X_test_original[artificial_mask] - X_test_imputed[artificial_mask]) ** 2))
                cv_errors.append(error)
        
        if cv_errors:
            mean_error = np.mean(cv_errors)
            if mean_error < min_error:
                min_error = mean_error
                best_k = k
    
    if verbose:
        print(f"Best k: {best_k}, RMSE: {min_error:.4f}")
    
    # Final imputation with best k
    final_imputer = KNNImputer(n_neighbors=best_k)
    X_imputed = final_imputer.fit_transform(X)
    
    return X_imputed, best_k


def identify_variable_types(df):
    """
    Identify variable types in clinical data
    Returns:
    --------
    binary_cols: list of binary columns (0,1) - typically yes/no clinical indicators
    ordinal_cols: list of ordinal columns (1,2,3...) - typically severity scales
    continuous_cols: list of continuous columns - typically measurements
    """
    binary_cols = []
    ordinal_cols = []
    continuous_cols = []
    
    for col in df.columns:
        unique_vals = sorted(df[col].dropna().unique())
        n_unique = len(unique_vals)
        
        # Check if values are numeric
        try:
            numeric_vals = [float(x) for x in unique_vals]
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
            continue
            
        if not is_numeric:
            continue
            
        # Binary variables (0,1)
        if n_unique == 2 and set(unique_vals).issubset({0, 1}):
            binary_cols.append(col)
        # Ordinal variables - look for integer sequences with small gaps
        elif n_unique <= 10 and all(isinstance(x, (int, np.integer)) or x.is_integer() for x in numeric_vals):
            # Check if values form a reasonable sequence (e.g., 1,2,3 or 1,3,5)
            diffs = np.diff(numeric_vals)
            if np.all(diffs <= 2) and np.all(diffs > 0):  # Allow gaps up to 2
                ordinal_cols.append(col)
            else:
                continuous_cols.append(col)
        # Continuous variables
        else:
            continuous_cols.append(col)
            
    return binary_cols, ordinal_cols, continuous_cols


def process_imputation(X, n_splits=5, k_values=[3, 5, 7, 10], verbose=False):
    """
    Clinical data-aware imputation process
    
    Parameters:
    -----------
    X: pd.DataFrame or np.array
        Clinical data with missing values
    
    Returns:
    --------
    imputed_data: np.array
        Imputed data with preserved variable types
    best_k: int
        Best k value used for imputation
    """
    # Convert to DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # Identify variable types
    binary_cols, ordinal_cols, continuous_cols = identify_variable_types(X)
    if verbose:
        print("\nVariable types found:")
        print(f"Binary (yes/no) variables: {len(binary_cols)}")
        print(f"Ordinal (severity scales): {len(ordinal_cols)}")
        print(f"Continuous (measurements): {len(continuous_cols)}")
        if ordinal_cols:
            print("Ordinal columns:", ordinal_cols)
    
    # Store original ordinal values for each column
    ordinal_values = {}
    for col in ordinal_cols:
        ordinal_values[col] = sorted(X[col].dropna().unique())
    
    # Separate scaling for each type
    X_scaled = X.copy()
    scalers = {}
    
    # Scale continuous variables
    if continuous_cols:
        scalers['continuous'] = MinMaxScaler()
        X_scaled[continuous_cols] = scalers['continuous'].fit_transform(X[continuous_cols])
    
    # Scale ordinal variables to integers 0 to n-1
    if ordinal_cols:
        scalers['ordinal'] = {}
        for col in ordinal_cols:
            unique_vals = ordinal_values[col]
            val_map = {val: i for i, val in enumerate(unique_vals)}
            X_scaled[col] = X[col].map(val_map)
            scalers['ordinal'][col] = val_map
    
    # Binary variables are already 0-1 scaled
    
    # Perform CVBKNN imputation
    X_imputed_scaled, best_k = cvbknn_imputer(X_scaled.values, n_splits=n_splits, k_values=k_values, verbose=verbose)
    X_imputed = pd.DataFrame(X_imputed_scaled, columns=X.columns, index=X.index)
    
    # Restore original scales and types
    if continuous_cols:
        X_imputed[continuous_cols] = scalers['continuous'].inverse_transform(X_imputed[continuous_cols])
    
    # Restore ordinal variables
    for col in ordinal_cols:
        val_map = scalers['ordinal'][col]
        inv_map = {i: val for val, i in val_map.items()}
        # Round to nearest integer and map back
        X_imputed[col] = X_imputed[col].round().map(inv_map)
        # Ensure values are in valid set
        valid_values = np.array(ordinal_values[col])
        X_imputed[col] = pd.Series(X_imputed[col]).apply(
            lambda x: valid_values[np.abs(valid_values - x).argmin()]
        )
    
    # Round binary variables strictly to 0 or 1
    for col in binary_cols:
        X_imputed[col] = (X_imputed[col] > 0.5).astype(int)
    
    if verbose:
        print("\nImputation complete:")
        print(f"Best k value: {best_k}")
        print("Original data types and ranges preserved")
    
    return X_imputed.values, best_k


def test_imputation():
    """
    Test imputation on different variable types in random positions
    """
    import numpy as np
    import pandas as pd
    
    # Create test data with mixed types in random positions
    np.random.seed(42)
    n_samples = 100
    
    # Create data with missing values
    data = {
        'continuous_var': np.random.normal(50, 10, n_samples),  # Continuous
        'binary_var': np.random.choice([0, 1], n_samples),      # Binary
        'ordinal_var': np.random.choice([1, 2, 3], n_samples),  # Ordinal (1-3)
        'another_continuous': np.random.uniform(0, 100, n_samples),
        'another_binary': np.random.choice([0, 1], n_samples),
        'another_ordinal': np.random.choice([1, 2, 3, 4, 5], n_samples)  # Ordinal (1-5)
    }
    
    # Convert to DataFrame and introduce missing values
    df = pd.DataFrame(data)
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.2  # 20% missing values
        df.loc[mask, col] = np.nan
    
    # Store original ordinal values
    ordinal_cols = ['ordinal_var', 'another_ordinal']
    original_ordinal_values = {col: sorted(df[col].dropna().unique()) for col in ordinal_cols}
    
    # Impute data
    imputed_data, best_k = process_imputation(df, verbose=True)
    imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
    
    # Verify results
    print("\nVerification Results:")
    for col in df.columns:
        if 'binary' in col:
            assert set(imputed_df[col].unique()).issubset({0, 1}), f"Binary variable {col} contains non-binary values"
            print(f"✓ {col}: Binary values preserved (0/1 only)")
        elif 'ordinal' in col:
            valid_values = original_ordinal_values[col]
            assert all(v in valid_values for v in imputed_df[col].unique()), \
                f"Ordinal variable {col} contains invalid values. Found {sorted(imputed_df[col].unique())}, expected {valid_values}"
            print(f"✓ {col}: Ordinal values preserved {valid_values}")
        else:
            assert imputed_df[col].notna().all(), f"Continuous variable {col} contains missing values"
            print(f"✓ {col}: Continuous values properly imputed")
    
    return imputed_df


def process_features(df):
    """
    Drop highly correlated features
    """
    # Keep track of dropped features
    dropped_features = []
    
    # Drop highly correlated features
    original_shape = df.shape
    df = drop_correlated_features(df)
    
    # Update dropped features
    dropped_correlated = original_shape[1] - df.shape[1]
    if dropped_correlated > 0:
        dropped_features.append(f"Dropped {dropped_correlated} highly correlated features")
    
    return df, dropped_features


# Run test if module is run directly
if __name__ == "__main__":
    test_imputation()
