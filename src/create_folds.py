import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def create_stratified_folds(X, y, n_splits=5, random_state=42, min_fold_size=2):
    """
    Create stratified folds for regression target by binning the continuous values.
    
    Args:
        X: Feature matrix or dataframe
        y: Continuous target values
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        min_fold_size: Minimum samples per fold for each value
    
    Returns:
        Array of fold indices
    """
    # Remove NaN values
    valid_indices = ~np.isnan(y)
    y_valid = y[valid_indices]
    
    print(f"\nInitial data stats:")
    print(f"Total samples: {len(y)}")
    print(f"Valid samples: {len(y_valid)}")
    print(f"NaN samples: {len(y) - len(y_valid)}")
    print(f"Score range: {np.min(y_valid):.2f} to {np.max(y_valid):.2f}")
    print(f"Score mean: {np.mean(y_valid):.2f}, std: {np.std(y_valid):.2f}")
    
    # Count unique values and their frequencies
    unique_values, value_counts = np.unique(y_valid, return_counts=True)
    print("\nValue distribution:")
    for val, count in zip(unique_values, value_counts):
        print(f"Score {val:.1f}: {count} samples ({count/len(y_valid)*100:.1f}%)")
    
    # Calculate minimum samples needed for proper stratification
    min_samples = min_fold_size * n_splits
    
    # Group rare values together
    y_binned = y_valid.copy()
    
    # Find values with too few samples
    rare_mask = value_counts < min_samples
    # Don't consider a value rare if it has more than min_samples/2 samples
    rare_mask &= value_counts < min_samples/2
    rare_values = unique_values[rare_mask]
    
    if len(rare_values) > 0:
        print(f"\nFound {len(rare_values)} rare values with < {min_samples/2:.1f} samples:")
        for val in rare_values:
            count = np.sum(y_valid == val)
            print(f"Score {val:.1f}: {count} samples")
        
        # For each rare value, find the closest non-rare value
        for rare_val in rare_values:
            # Find closest non-rare value
            non_rare_values = unique_values[~rare_mask]
            closest_val = non_rare_values[np.argmin(np.abs(non_rare_values - rare_val))]
            print(f"Grouping score {rare_val:.1f} with {closest_val:.1f} for stratification")
            y_binned[y_valid == rare_val] = closest_val
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Create fold assignments array
    fold_indices = np.full(len(y), -1)
    
    # Assign folds only to valid indices
    valid_idx = np.where(valid_indices)[0]
    for fold, (_, test_idx) in enumerate(skf.split(np.zeros(len(y_valid)), y_binned)):
        fold_indices[valid_idx[test_idx]] = fold
        
        # Get actual indices for this fold
        fold_test_idx = valid_idx[test_idx]
        fold_train_idx = np.array([i for i in valid_idx if i not in fold_test_idx])
        
        # Print fold statistics
        print(f"\nFold {fold}:")
        print(f"Train size = {len(fold_train_idx)}, Test size = {len(fold_test_idx)}")
        
        # Print score distributions
        train_scores = y[fold_train_idx]
        test_scores = y[fold_test_idx]
        
        print(f"Train set score distribution:")
        print(f"Min: {np.min(train_scores):.2f}, Max: {np.max(train_scores):.2f}")
        print(f"Mean: {np.mean(train_scores):.2f}, Std: {np.std(train_scores):.2f}")
        
        print(f"Test set score distribution:")
        print(f"Min: {np.min(test_scores):.2f}, Max: {np.max(test_scores):.2f}")
        print(f"Mean: {np.mean(test_scores):.2f}, Std: {np.std(test_scores):.2f}")
        
        # Print value distribution in train and test sets
        for val in unique_values:
            train_count = np.sum(train_scores == val)
            test_count = np.sum(test_scores == val)
            print(f"Score {val:.1f} - Train: {train_count} ({train_count/len(train_scores)*100:.1f}%), "
                  f"Test: {test_count} ({test_count/len(test_scores)*100:.1f}%)")
    
    return fold_indices

if __name__ == "__main__":
    # Load data
    df = pd.read_excel("../input/df_merged_v2.xlsx")
    
    # Test both score types
    for score_type in ["FRIED", "FRAGIRE18"]:
        print(f"\n{'='*50}")
        print(f"Creating folds for {score_type}")
        print(f"{'='*50}")
        
        # Set target column based on score type
        target_col = "Fried_Score_FRIED_TOTAL_Version_1" if score_type == "FRIED" else "Frailty_Score_FRAGIRE18_SQ001"
        
        # Create folds
        fold_indices = create_stratified_folds(
            X=df,
            y=df[target_col].values,
            n_splits=5,
            random_state=42,
            min_fold_size=2
        )
        
        # Add fold column to dataframe
        fold_col = f'kfold_{score_type.lower()}'
        df[fold_col] = fold_indices
    
    # Save the updated dataset with both fold columns
    df.to_csv("../input/train_folds.csv", index=False)
