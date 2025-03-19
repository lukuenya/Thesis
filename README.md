# PROJECT OVERVIEW
This project is for my master thesis in Data Science. The aim is to develop a machine learning model that (predicts frailty scores - For Regression) or classifies participants as healthy or frail (For Classification) based on both the "FRIED protocol" and the "Expert's Eye". 

The focus is on understanding the features driving these scores or classes (feature importance), comparing the two scoring systems (Expert's Eye vs. FRIED), and identifying their common feature space to improve explainability and clinical utility.

For easiness purposes, we will use binary classification problem to predict the FRIED and FRAGIRE18(Expert's Eye) states of frailty: 0 for non-frail and 1 for frail. 

## Research Objectives
1. Develop accurate prediction models for both FRIED and FRAGIRE18 frailty score/class
2. Identify key features that influence both scoring systems(feature importance)
3. Compare and contrast these two frailty assessment methods
4. Evaluate the clinical applicability and reliability of ML-based frailty assessments

# METHODOLOGY

## Target Variables
- Classification (binary: 0 for healthy and 1 for frail)
   - Score type: FRIED -> 'Fried_State'
   - Score type: FRAGIRE18 -> 'Frailty_State_GFST'
- Regression (continuous: score value) : NOT USED IN THIS PROJECT
   - Score type: FRIED -> 'Fried_Score_FRIED_TOTAL_Version_1'
   - Score type: FRAGIRE18 -> 'Frailty_Score_FRAGIRE18_SQ001'

## Data Preprocessing
1. Feature Selection and Engineering
   - Removal of highly correlated features (list in config.py)
   - Missing values :
      - Option 1 : handled in 'X' by models (if applicable) but in 'y' respective rows are dropped.
      - Option 2 : imputed by the Cross-validation based KNN imputer.
   - Class Imbalance :
      - Option 1 : handled by models (if applicable) by adjusting class weights.
      - Option 2 : SMOTE + Tomek Links.
      - Option 3 : Holdout method (80% train, 20% test) with random resampling.


2. Data Splitting
   - Stratified k-fold cross-validation
   - 75% of the data is used for training
      - 100 samples for FRIED
      - 108 samples for FRAGIRE18
   - 25% of the data is used for validation (unseen during training)
      - 32 samples for FRIED
      - 36 samples for FRAGIRE18

## Model Development

### Models Evaluated
1. LightGBM
2. XGBoost
3. CatBoost
4. Random Forest

### Hyperparameter Optimization
- Framework: Optuna
- Optimization metric: 
   - Classification: ROC AUC
- Number of trials: 15
- Cross-validation: 5-fold

## Validation Framework

### Statistical Validation

1. **Model Performance Metrics**
   - Classification:
     - ROC AUC
     - ROC curve
     - Precision-Recall curve

2. **Cross-Validation Strategy**
   - 5-fold Cross-validation
      - Classification: Stratified by class distribution

### Clinical Validation
 - Feature Importance
 - Expert Evaluation

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

- `models/`: Trained models and parameters
  - Saved model for later use in notebooks

- `notebooks/`: Jupyter notebooks
  - Exploratory data analysis
  - Feature engineering experiments
  - Result visualizations

# RESULTS INTERPRETATION

## Clinical Impact Assessment
- Feature importance for clinical interpretation

## Score System Comparison
- Correlation between FRIED and FRAGIRE18
- Common influential features
- Complementary aspects of both systems

# REPRODUCIBILITY
All experiments can be reproduced using the provided code:
1. Configure parameters in `config.py`
2. Run feature importance analysis:
   ```bash
   cd src
   python get_feature_importances.py --score_type FRIED --model_name [model] --task [task]
   python get_feature_importances.py --score_type FRAGIRE18 --model_name [model] --task [task]
   ```
   - Replace `[model]` with: `lightgbm`, `xgboost`, `randomforest` or `catboost`

3. Run feature selection with desired method:
   ```bash
   # Using embedded method (threshold-based)
   python feature_selection.py --score_type FRIED --method embedded --threshold_percentile 20
   
   # Using wrapper method with specific number of features
   python feature_selection.py --score_type FRIED --method wrapper --n_features 10
   
   # Without imputation (using raw data)
   python feature_selection.py --score_type FRIED --method wrapper --n_features 10 --no_imputation
   ```

4. Train and optimize models:
   ```bash
   # Using embedded feature selection
   python train_optimize.py --score_type FRIED --model [model] --feature_selection embedded
   
   # Using wrapper feature selection
   python train_optimize.py --score_type FRIED --model [model] --feature_selection wrapper
   
   # Using all features (no feature selection)
   python train_optimize.py --score_type FRIED --model [model] --no_feature_selection
   
   # Without imputation
   python train_optimize.py --score_type FRIED --model [model] --no_imputation
   ```

The output files will be organized in folders based on the processing steps (imputation and feature selection method).

# DEPENDENCIES
Required Python packages are listed in `requirements.txt`