# Model Analysis Documentation

## Data Preprocessing

### Feature Selection
- All features are preprocessed to ensure compatibility with ML models
- Column names are cleaned by replacing special characters with underscores
- Missing values are handled through [specify your method]
- Outliers are [specify your handling method]

### Fold Creation Strategy
The dataset is split using a stratified k-fold cross-validation approach (`create_folds.py`):

1. **Basic Setup:**
   - Number of folds: 5
   - Random seed: 42 (for reproducibility)
   - Minimum fold size: 2 samples per score value

2. **Data Preprocessing:**
   - Handles NaN values in target scores
   - Reports initial data statistics:
     - Total sample count
     - Valid sample count
     - Score range, mean, and standard deviation
     - Distribution of score values

3. **Stratification Strategy:**
   - Identifies rare score values (< min_fold_size * n_splits samples)
   - Groups rare values with their closest non-rare neighbors
   - Ensures balanced distribution of scores across folds
   - Maintains clinical relevance in grouping decisions

4. **Fold Statistics:**
   For each fold:
   - Reports train/test set sizes
   - Shows score distribution statistics (min, max, mean, std)
   - Provides detailed value distribution in both train and test sets
   - Validates stratification effectiveness

5. **Separate Folds for Each Score:**
   - Creates independent stratified folds for FRIED and FRAGIRE18 scores
   - Saves fold assignments in separate columns:
     - `kfold_fried` for FRIED score
     - `kfold_fragire18` for FRAGIRE18 score

6. **Output:**
   - Saves the processed dataset with fold assignments to `train_folds.csv`
   - Maintains all original features and adds fold columns
   - Enables consistent cross-validation across all models

### Class Handling
- FRIED Score (1-5 scale):
  - Score 3 and above indicates severe cases
  - Class weights are applied to handle imbalance
- FRAGIRE18 Score (1-10 scale):
  - Class weights are applied to handle imbalance

## Model Training

### Cross-Validation
- Stratified k-fold cross-validation (k=5)
- Fold 4 is held out as test set
- Same folds used across all models for fair comparison

### Hyperparameter Optimization
- Optimization using Optuna framework
- Number of trials: [specify]
- Optimization metric: RMSE
- Early stopping: [specify criteria]

### Model Architectures
1. LightGBM
   - Hyperparameter ranges: [specify]
   - Key parameters: [specify]

2. XGBoost
   - Hyperparameter ranges: [specify]
   - Key parameters: [specify]

3. CatBoost
   - Hyperparameter ranges: [specify]
   - Key parameters: [specify]

4. TabNet
   - Hyperparameter ranges: [specify]
   - Key parameters: [specify]

## Evaluation Metrics

### Primary Metrics
- Root Mean Square Error (RMSE)
  - Both weighted and unweighted
  - Confidence intervals via bootstrapping
- Mean Absolute Error (MAE)
  - Per-class analysis
  - Confidence intervals via bootstrapping

### Clinical Significance
- FRIED Score Thresholds:
  - Minimal: 0.5 points
  - Moderate: 1.0 points
  - Critical: 2.0+ points

- FRAGIRE18 Score Thresholds:
  - Minimal: 1.0 points
  - Moderate: 2.0 points
  - Critical: 3.0+ points

### Model Calibration
- Analysis using binned predictions
- Calibration plots with error bars
- Per-bin statistics

## Score Comparison Analysis

### Correlation Analysis
- Correlation between true scores
- Correlation between predicted scores
- Visualization of score relationships
- Analysis of prediction agreement

### Feature Importance Comparison
- Common important features between scores
- Relative importance analysis
- Top 20 features comparison
- Feature overlap visualization

### Clinical Impact Analysis
- Analysis of critical prediction errors
- Dangerous misses identification:
  - FRIED: Cases where severity is underestimated
  - FRAGIRE18: Cases where health is overestimated
- Per-score error distribution

## Limitations and Future Work

### Current Limitations
1. Internal validation only
   - Need external validation dataset
   - Or temporal validation

2. Clinical Validation
   - Thresholds need clinical expert validation
   - Impact analysis needs clinical context

### Future Improvements
1. External Validation
   - Collect independent test dataset
   - Implement temporal validation

2. Clinical Integration
   - Collaborate with clinical experts
   - Validate error thresholds
   - Assess impact on clinical decisions

3. Model Enhancements
   - Ensemble methods
   - Deep learning architectures
   - Uncertainty quantification
