"""
Functions to format longitudinal data for analysis and machine learning.
"""
import pandas as pd


def create_long_format(df, id_col='Foldername', date_col='bilan date'):
    """
    Create long format dataset with baseline/follow-up indicator.
    Single measurement participants are placed at the end.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with longitudinal data
    id_col : str, default='Foldername'
        Column name containing participant IDs
    date_col : str, default='bilan date'
        Column name containing the measurement dates
    
    Returns:
    --------
    pandas.DataFrame
        Long format dataset with measurement_type indicator
    """
    # Sort by Foldername and date
    df_sorted = df.sort_values([id_col, date_col])
    
    # Count measurements per Foldername
    measurement_counts = df_sorted.groupby(id_col).size()
    
    # Create measurement type indicator
    df_sorted['measurement_type'] = df_sorted.groupby(id_col).cumcount().map({0: 'baseline', 1: 'follow-up'})
    
    # Identify single Foldernames
    single_measurement = measurement_counts[measurement_counts == 1].index
    multiple_measurement = measurement_counts[measurement_counts > 1].index
    
    # Split dataframe
    df_multiple = df_sorted[df_sorted[id_col].isin(multiple_measurement)]
    df_single = df_sorted[df_sorted[id_col].isin(single_measurement)]
    
    # Concatenate with single measurements at the end
    df_long = pd.concat([df_multiple, df_single])
    
    # Add number of measurements column
    df_long['n_measurements'] = df_long[id_col].map(measurement_counts)
    
    return df_long


def create_wide_format(df, id_col='Foldername', date_col='bilan date'):
    """
    Create wide format dataset with baseline and follow-up side by side.
    Also computes change scores and adds indicators for data completeness.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with longitudinal data
    id_col : str, default='Foldername'
        Column name containing participant IDs
    date_col : str, default='bilan date'
        Column name containing the measurement dates
    
    Returns:
    --------
    pandas.DataFrame
        Wide format dataset with baseline and follow-up measurements side by side
    """
    # First create long format to ensure proper ordering
    df_long = create_long_format(df, id_col, date_col)
    
    # Get columns to pivot (exclude created columns)
    value_columns = [col for col in df_long.columns 
                    if col not in [id_col, 'measurement_type', 'n_measurements']]
    
    # Pivot to wide format
    df_wide = df_long.pivot(
        index=id_col,
        columns='measurement_type',
        values=value_columns
    )
    
    # Flatten column names
    df_wide.columns = [f'{col[0]}_{col[1]}' for col in df_wide.columns]
    df_wide = df_wide.reset_index()
    
    # Add indicators
    df_wide['has_followup'] = ~df_wide[[col for col in df_wide.columns 
                                       if col.endswith('_follow-up')]].isna().all(axis=1)
    
    # Compute change scores for numeric columns
    baseline_cols = [col for col in df_wide.columns if col.endswith('_baseline')]
    for col in baseline_cols:
        base_name = col.replace('_baseline', '')
        followup_col = f'{base_name}_follow-up'
        
        # Only compute change if both columns exist and are numeric, excluding bilan_id columns
        if (followup_col in df_wide.columns and 
            pd.api.types.is_numeric_dtype(df_wide[col]) and 
            pd.api.types.is_numeric_dtype(df_wide[followup_col]) and
            'bilan_id' not in base_name):  
            df_wide[f'{base_name}_change'] = df_wide[followup_col] - df_wide[col]
    
    return df_wide
