# PROJECT OVERVIEW
This project is for my master thesis in Data Science. The aim is to develop a machine learning model that (predicts frailty scores - For Regression) or classifies participants as healthy or frail (For Classification) based on both the "FRIED protocol" and the "Expert's Eye". 
The focus is on understanding the features driving these scores or classes, comparing the two scoring systems or classification models, and identifying their common feature space to improve explainability and clinical utility.
For easiness purposes, we will use binary classification problem to predict the FRIED and FRAGIRE18 states of frailty: 0 for healthy and 1 for frail. 

## Research Objectives
1. Develop accurate prediction models for both FRIED and FRAGIRE18 frailty score
2. Compare and contrast these two frailty assessment methods
3. Identify key features that influence both scoring systems or classification models
4. Evaluate the clinical applicability and reliability of ML-based frailty assessments

# METHODOLOGY

## Data Preprocessing
1. Feature Selection and Engineering
   - Removal of highly correlated features (list in config.py)
   - Missing values are handled natively by models


2. Data Splitting
   - Stratified k-fold cross-validation
   - 80% of the data is used for training
   - 20% of the data is used for validation

## Model Development

### Models Evaluated
1. LightGBM
   - Gradient boosting framework
   - Leaf-wise tree growth

2. XGBoost
   - Regularized gradient boosting
   - Built-in handling of missing values
   - Second-order gradients

3. CatBoost
   - Ordered boosting
   - Reduced overfitting

### Hyperparameter Optimization
- Framework: Optuna
- Optimization metric: AUC and F1-Score
- Number of trials: 100
- Cross-validation: 5-fold

## Validation Framework

### Statistical Validation

1. **Model Performance Metrics**
   - ROC AUC
   - ROC curve
   - Precision-Recall curve

2. **Bootstrap Confidence Intervals**
   - Methodology:
     * 1000 bootstrap samples with replacement
     * Preserves data distribution
   - Metrics Computed:
     * 95% CI 
     * 

3. **Error Distribution Analysis**


4. **Cross-Validation Strategy**
   - 5-fold Cross-validation
     * Stratified by score distribution

### Clinical Validation

### Future Enhancements
Potential areas for extending the validation framework:
1. Additional Statistical Metrics


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
      - data_global.xlsx

- `src/`: Python source code
  - `train_optimize.py`: Model training and optimization
  - `get_feature_importances.py`: Feature importance analysis
  - `feature_selection.py`: Feature selection and aggregation
  - `config.py`: Configuration file
  - `data_loader.py`: Data loading
  - `preprocessing.py`: Data preprocessing

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
- ROC AUC with 95% confidence intervals

## Clinical Impact Assessment
- Feature importance for clinical interpretation

## Score System Comparison
- Correlation between FRIED and FRAGIRE18
- Common influential features
- Complementary aspects of both systems

# REPRODUCIBILITY
All experiments can be reproduced using the provided code:
1. Configure parameters in `config.py`
2. Run :
      - `python src/get_feature_importances.py --score_type FRIED --model_name lightgbm`
      - `python src/get_feature_importances.py --score_type FRIED --model_name xgboost`
      - `python src/get_feature_importances.py --score_type FRIED --model_name catboost`
      - `python src/get_feature_importances.py --score_type FRAGIRE18 --model_name lightgbm`
      - `python src/get_feature_importances.py --score_type FRAGIRE18 --model_name xgboost`
      - `python src/get_feature_importances.py --score_type FRAGIRE18 --model_name catboost`
   for dimensionality reduction based on each model feature importances.
3. Run :
      - `python feature_selection.py --score_type FRIED --threshold_percentile 20`
      - `python feature_selection.py --score_type FRAGIRE18 --threshold_percentile 20`
   to aggregate feature importances and select top features for each score type.
4. Train final models with selected features:
      - `python train_optimize.py --score_type FRIED --model_name lightgbm --selected_features`
      - `python train_optimize.py --score_type FRAGIRE18 --model_name lightgbm --selected_features`

# DEPENDENCIES
Required Python packages are listed in `requirements.txt`