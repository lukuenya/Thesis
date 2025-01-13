# PROJECT OVERVIEW
This project is for my master thesis in Data Science. The aim is to develop a machine learning model that predicts frailty scores based on both the FRIED protocol and the "Expert's Eye." The focus is on understanding the features driving these scores, comparing the two scoring systems, and identifying their common feature space to improve explainability and clinical utility.
For easiness purposes, we will use binary classification problem to predict the FRIED and FRAGIRE18 states of frailty: 0 for healthy and 1 for frail. 

## Research Objectives
1. Develop accurate prediction models for both FRIED and FRAGIRE18 frailty score
2. Compare and contrast these two frailty assessment methods
3. Identify key features that influence both scoring systems
4. Evaluate the clinical applicability and reliability of ML-based frailty assessment

## Clinical Significance
- FRIED Score (1-5, higher is worse): Standard frailty assessment protocol
  - Score grouping: 4.0 grouped with 3.0 due to rare occurrences
  - Critical threshold: 2-point difference (may miss severe cases)
  
- FRAGIRE18 Score (1-10, higher is better): Expert-based assessment
  - Score grouping: 4.0 grouped with 5.0 due to rare occurrences
  - Critical threshold: 3-point difference

# METHODOLOGY

## Data Preprocessing
1. Feature Selection and Engineering
   - Removal of highly correlated features
   - Handling of missing values
   - Feature normalization
   - Categorical variable encoding

2. Data Splitting
   - Stratified k-fold cross-validation
   - Preservation of patient groups in splits
   - Balanced representation of score distributions

## Model Development

### Models Evaluated
1. LightGBM
   - Gradient boosting framework
   - Efficient handling of categorical features
   - Leaf-wise tree growth

2. XGBoost
   - Regularized gradient boosting
   - Built-in handling of missing values
   - Second-order gradients

3. CatBoost
   - Ordered boosting
   - Advanced categorical feature handling
   - Reduced overfitting

### Hyperparameter Optimization
- Framework: Optuna
- Optimization metric: RMSE
- Number of trials: 100
- Cross-validation: 5-fold

## Validation Framework

### Statistical Validation

1. **Model Performance Metrics**
   - Root Mean Square Error (RMSE)
     * Primary metric for model comparison
     * Penalizes larger errors more heavily
     * Used for model selection and comparison
   - Mean Absolute Error (MAE)
     * Secondary metric for interpretability
     * Direct measure of average prediction deviation

2. **Bootstrap Confidence Intervals**
   - Methodology:
     * 1000 bootstrap samples with replacement
     * Preserves data distribution
   - Metrics Computed:
     * 95% CI for RMSE: Robust estimate of model error range
     * 95% CI for MAE: Range of expected prediction deviation
     * Standard deviation of metrics

3. **Error Distribution Analysis**
   - Basic Statistics:
     * Mean error: Central tendency of predictions
     * Median error: Robust measure of central tendency
     * Standard deviation: Spread of prediction errors
   - Percentile Analysis:
     * 95th percentile errors: Worst-case scenario bounds

4. **Cross-Validation Strategy**
   - 5-fold Cross-validation
     * Stratified by score distribution
     * Preserves patient groups
     * Ensures representative splits

### Clinical Validation

1. **Error Threshold Analysis**
   
   a) FRIED Score (1-5, higher indicates worse condition)
   - Minimal Impact (±0.5 points)
     * Acceptable for routine screening
   - Moderate Impact (±1.0 point)
     * May affect clinical decisions
   - Critical Impact (±2.0 points)
     * Risk of missed severe cases

   b) FRAGIRE18 Score (1-10, higher indicates better condition)
   - Minimal Impact (±1.0 point)
     * Within normal assessment variation
   - Moderate Impact (±2.0 points)
     * Potential impact on care decisions
   - Critical Impact (±3.0 points)
     * Risk of inappropriate care level

2. **Risk Analysis**
   - Critical Error Types:
     * FRIED: Underestimating frailty severity
     * FRAGIRE18: Overestimating health status
   - Impact Metrics:
     * Percentage of predictions in each error category
     * Identification of dangerous misses

3. **Score System Comparison**
   - Correlation Analysis:
     * Pearson correlation between true scores
     * Correlation between predicted scores
   - Feature Analysis:
     * Common important features between systems
     * System-specific important features

### Future Enhancements
Potential areas for extending the validation framework:
1. Additional Statistical Metrics
   - R-squared score for model explanation
   - Advanced error distribution analysis (skewness, kurtosis)
   - Comprehensive outlier detection

2. Enhanced Clinical Validation
   - Prediction confidence scoring
   - Decision boundary analysis
   - Systematic bias detection

3. Implementation Considerations
   - Real-time prediction capability
   - Integration requirements
   - Quality assurance protocols

# PROJECT STRUCTURE
- `input/`: Input files and data
  - Training data with fold assignments
  - Feature descriptions
  - Score definitions

- `src/`: Python source code
  - `optuna_optimization.py`: Model training and optimization
  - `analyze_optimization.py`: Result analysis and validation
  - `config.py`: Configuration parameters
  - `utils.py`: Utility functions

- `models/`: Trained models and optimization studies
  - Saved model states
  - Optuna study objects
  - Model parameters

- `notebooks/`: Jupyter notebooks
  - Exploratory data analysis
  - Feature engineering experiments
  - Result visualizations

# RESULTS INTERPRETATION

## Model Performance Metrics
- RMSE with 95% confidence intervals
- Mean and median prediction errors
- Per-class performance analysis

## Clinical Impact Assessment
- Percentage of predictions within acceptable thresholds
- Analysis of potentially dangerous predictions
- Feature importance for clinical interpretation

## Score System Comparison
- Correlation between FRIED and FRAGIRE18
- Common influential features
- Complementary aspects of both systems

# REPRODUCIBILITY
All experiments can be reproduced using the provided code:
1. Configure parameters in `config.py`
2. Run model optimization: `python src/optuna_optimization.py`
3. Analyze results: `python src/analyze_optimization.py`

# DEPENDENCIES
Required Python packages are listed in `requirements.txt`