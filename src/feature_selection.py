import pandas as pd
import numpy as np
import joblib
import os
import data_loader
import config
import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def load_feature_importances(score_type, imputation=True, feature_selection=None):
    """
    Load feature importance files for all models
    
    Parameters:
    -----------
    score_type : str
        Score type (e.g., 'FRIED', 'FRAGIRE18')
    imputation : bool
        Whether imputation is used
    feature_selection : str or None
        Feature selection method used before (to maintain path consistency)
    """
    importances = {}
    models = ['lightgbm', 'xgboost', 'catboost', 'randomforest']
    
    # Get output directory based on processing steps
    paths = config.get_output_paths(imputation, feature_selection)
    output_dir = paths['feature_importances']
    
    for model in models:
        # Try the CSV format that get_feature_importances.py creates
        filepath = os.path.join(
            output_dir,
            f"importance_{model}_{score_type.lower()}_classification.csv"
        )
        if os.path.exists(filepath):
            print(f"Loading feature importances from {filepath}")
            df = pd.read_csv(filepath)
            # Rename columns if needed to match expected format
            if 'feature' in df.columns and 'importance' in df.columns:
                df = df.rename(columns={
                    'feature': 'Feature',
                    'importance': 'Importance'
                })
                # Add standard deviation column if missing
                if 'Importance_STD' not in df.columns:
                    df['Importance_STD'] = df['Importance'] * 0.1  # Default value
            importances[model] = df
    
    return importances


def aggregate_feature_importance(importances):
    """
    Aggregate feature importances from different models
    """
    # Initialize dictionary to store normalized scores
    normalized_importances = {}
    scaler = MinMaxScaler()
    
    for model_name, imp_df in importances.items():
        # Reshape for scaler and normalize importance scores
        importance_scores = imp_df['Importance'].values.reshape(-1, 1)
        normalized_scores = scaler.fit_transform(importance_scores).ravel()
        
        # Scale standard deviations using the same scaling factor
        # This preserves the relative uncertainty while keeping the scale consistent
        scale_factor = (scaler.data_max_ - scaler.data_min_)[0]
        normalized_stds = imp_df['Importance_STD'].values / scale_factor if scale_factor != 0 else imp_df['Importance_STD'].values
        
        normalized_importances[model_name] = pd.DataFrame({
            'Feature': imp_df['Feature'],
            'Normalized_Importance': normalized_scores,
            'Normalized_STD': normalized_stds
        })
    
    # Calculate mean and std across models for each feature
    all_features = normalized_importances[list(normalized_importances.keys())[0]]['Feature']
    mean_importance = pd.DataFrame({
        'Feature': all_features,
        'Mean_Importance': 0.0,
        'Std_Importance': 0.0
    })
    
    for feature in all_features:
        feature_scores = []
        for model_scores in normalized_importances.values():
            score = model_scores[model_scores['Feature'] == feature]['Normalized_Importance'].values[0]
            feature_scores.append(score)
        
        mean_importance.loc[mean_importance['Feature'] == feature, 'Mean_Importance'] = np.mean(feature_scores)
        mean_importance.loc[mean_importance['Feature'] == feature, 'Std_Importance'] = np.std(feature_scores)
    
    return mean_importance.sort_values('Mean_Importance', ascending=False)


def select_top_features(agg_importance, threshold_percentile=20):
    """
    Select top features based on importance scores
    threshold_percentile: Keep features with importance in top X percentile
    """
    threshold = np.percentile(agg_importance['Mean_Importance'], 100 - threshold_percentile)
    selected_features = agg_importance[agg_importance['Mean_Importance'] >= threshold]['Feature'].tolist()
    return selected_features


def save_selected_features(selected_features, score_type, imputation=True, feature_selection=None, prefix=''):
    """
    Save selected features to a file
    
    Parameters:
    -----------
    selected_features : list
        List of selected feature names
    score_type : str
        Score type (e.g., 'FRIED', 'FRAGIRE18')
    imputation : bool
        Whether imputation is used
    feature_selection : str or None
        Feature selection method used before (to maintain path consistency)  
    prefix : str
        Prefix for the filename (default: '')
    """
    # Get output directory based on processing steps
    paths = config.get_output_paths(imputation, feature_selection)
    output_dir = paths['feature_importances']
    
    filepath = os.path.join(
        output_dir,
        f"{prefix}selected_features_{score_type.lower()}_classification.txt"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    
    print(f"Selected features saved to: {filepath}")


def wrapper_feature_selection(X, y, n_features=10, random_state=42):
    """
    Wrapper method for feature selection using LightGBM for classification
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    n_features : int
        Number of features to select
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    feature_importances : pd.DataFrame
        DataFrame with feature importances
    """
    print(f"Starting wrapper-based feature selection with LightGBM to select {n_features} features...")
    
    # Initialize model for classification
    model = LGBMClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    # Initialize RFE with the model
    rfe = RFE(
        estimator=model,
        n_features_to_select=n_features,
        step=0.1  # Remove 10% of features at each iteration
    )
    
    # Fit RFE
    rfe.fit(X, y)
    
    # Get selected feature names
    feature_names = X.columns
    selected_features = feature_names[rfe.support_].tolist()
    
    # Get feature rankings
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    })
    
    # Sort by ranking
    feature_importances = feature_importances.sort_values('Ranking')
    
    print(f"Selected {len(selected_features)} features using wrapper method with LightGBM")
    print(f"Top 5 features: {selected_features[:5]}")
    
    return selected_features, feature_importances


def run_wrapper_selection(X, y, score_type, imputation=True, n_features=10):
    """
    Run wrapper-based feature selection and save results
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    score_type : str
        Score type (e.g., 'FRIED', 'FRAGIRE18')
    imputation : bool
        Whether imputation is used
    n_features : int
        Number of features to select
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    # Get output directory based on processing steps
    paths = config.get_output_paths(imputation, 'wrapper')
    output_dir = paths['feature_importances']
    
    # Run wrapper feature selection
    selected_features, feature_importances = wrapper_feature_selection(
        X, y, n_features=n_features
    )
    
    # Save feature importances to Excel
    os.makedirs(output_dir, exist_ok=True)
    feature_importances.to_excel(
        os.path.join(output_dir, f"wrapper_feature_importances_{score_type.lower()}.xlsx"),
        index=False
    )
    
    # Save selected features
    save_selected_features(
        selected_features, 
        score_type,
        imputation=imputation,
        feature_selection='wrapper',
        prefix="wrapper_"
    )
    
    return selected_features


def main(score_type, threshold_percentile=20, imputation=True):
    """
    Main function to perform embedded feature selection
    
    Parameters:
    -----------
    score_type : str
        Score type (e.g., 'FRIED', 'FRAGIRE18')
    threshold_percentile : int
        Percentile threshold for feature selection (1-100)
    imputation : bool
        Whether imputation is used
        
    Returns:
    --------
    selected_features : list
        List of selected feature names
    """
    # Load feature importances for all models
    importances = load_feature_importances(score_type, imputation)
    
    if not importances:
        print(f"No feature importance files found for {score_type}")
        return []
    
    # Aggregate importance scores across models
    agg_importance = aggregate_feature_importance(importances)
    
    # Select top features
    selected_features = select_top_features(agg_importance, threshold_percentile)
    
    # Save selected features
    save_selected_features(
        selected_features, 
        score_type, 
        imputation=imputation,
        feature_selection='embedded'
    )
    
    # Get output directory for plots
    paths = config.get_output_paths(imputation, 'embedded')
    output_dir = paths['feature_importances']
    
    # Plot feature importances
    plt.figure(figsize=(6, 8), dpi=300)  # High resolution
    
    # Sort features by mean importance
    sorted_agg = agg_importance.sort_values(by='Mean_Importance', ascending=False).reset_index(drop=True)
    
    # Plot top 30 features (or all if less than 30)
    n_features = min(30, len(sorted_agg))
    subset = sorted_agg.iloc[:n_features]
    
    # Plot with error bars
    plt.errorbar(
        x=range(n_features),
        y=subset['Mean_Importance'],
        yerr=subset['Std_Importance'],
        fmt='o',
        capsize=4,
        ecolor='red',
        markersize=8,
        markerfacecolor='blue',
        markeredgecolor='black'
    )
    
    plt.xticks(range(n_features), subset['Feature'], rotation=90)
    plt.ylabel('Normalized Importance')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"feature_importance_{score_type.lower()}.png"))
    plt.close()
    
    return selected_features


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Feature selection for classification')
    parser.add_argument('--score_type', type=str, default='FRIED',
                        help='Score to predict: FRIED or FRAGIRE18')
    parser.add_argument('--threshold', type=int, default=20,
                        help='Percentile threshold for embedded method (1-100)')
    parser.add_argument('--method', type=str, default='embedded',
                        help='Feature selection method: embedded or wrapper')
    parser.add_argument('--n_features', type=int, default=10,
                        help='Number of features to select for wrapper method')
    parser.add_argument('--no_imputation', action='store_true',
                        help='Use raw data without imputation')
    
    args = parser.parse_args()
    
    # Determine if imputation is used
    imputation = not args.no_imputation
    
    if args.method == 'embedded':
        selected_features = main(args.score_type, args.threshold, imputation)
        print(f"Selected {len(selected_features)} features with embedded method")
        print(f"Imputation: {'Yes' if imputation else 'No'}")
    elif args.method == 'wrapper':
        # Load data with or without imputation
        X, y = data_loader.load_data(target_score=args.score_type, imputation=imputation)
        
        # Run wrapper-based feature selection
        selected_features = run_wrapper_selection(
            X, y,
            score_type=args.score_type,
            imputation=imputation,
            n_features=args.n_features
        )
        print(f"Selected {len(selected_features)} features with wrapper method")
        print(f"Imputation: {'Yes' if imputation else 'No'}")
    else:
        print(f"Unknown method: {args.method}. Choose 'embedded' or 'wrapper'.")
