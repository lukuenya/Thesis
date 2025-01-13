# Model Analysis Documentation - Classification Models

## Data Preprocessing

### Feature Selection
- All features are preprocessed to ensure compatibility with ML models
- Column names are cleaned by replacing special characters with underscores
- Missing values are handled natively by models (if applicable)

### Data Splitting
- 80% of the data is used for training
- 20% of the data is used for validation


### Class Handling
- FRIED Score (1-5 scale):
  - Score 3 and above indicates severe cases
  - Classification column is called "Fried_State"

- FRAGIRE18 Score (4-10 scale):
  - Score 7 and above indicates the opposite of FRIED, indicating better health.
  - Classification column is called "Frailty_State_GFST"



## Model Training

### Cross-Validation
- Stratified k-fold cross-validation

### Hyperparameter Optimization
- Optimization using Optuna framework
- Number of trials: [specify]
- Optimization metric: ROC AUC
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
- ROC AUC

### Model Calibration
- 

## Score Comparison Analysis

### Correlation Analysis
-

### Feature Importance Comparison
- Common important features between scores
- Top 10 features comparison
- Feature overlap visualization

### Clinical Impact Analysis
- Analysis of critical prediction errors
- Dangerous misses identification:
  - FRIED: Cases where severity is underestimated
  - FRAGIRE18: Cases where health is overestimated
- 

## Limitations and Future Work

### Current Limitations
1. Internal validation only
   - Need external validation dataset
   - Or temporal validation

2. Clinical Validation


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
