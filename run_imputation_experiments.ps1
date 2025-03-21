# Define path to the results file
$resultsFile = "p:\Thesis\imputation_results.txt"
$srcDir = "p:\Thesis\src"

# Clear or create the results file
"Results for imputation experiments" | Out-File -FilePath $resultsFile -Force

# Define models and feature selection methods
$models = @("lightgbm", "xgboost", "catboost", "randomforest")
$featureSelectionMethods = @("embedded", "wrapper")

# Function to run feature importances for all models
function Run-FeatureImportances {
    param (
        [string]$scoreType
    )
    
    foreach ($model in $models) {
        # Make sure we're in the src directory
        Set-Location $srcDir
        
        $command = "python get_feature_importances.py --score_type $scoreType --model_name $model"
        Write-Host "Running: $command" -ForegroundColor Green
        Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    }
}

# Function to run feature selection
function Run-FeatureSelection {
    param (
        [string]$scoreType,
        [string]$method
    )
    
    # Make sure we're in the src directory
    Set-Location $srcDir
    
    # For embedded method, use threshold 10
    if ($method -eq "embedded") {
        $command = "python feature_selection.py --score_type $scoreType --method $method --threshold 10"
        Write-Host "Running: $command" -ForegroundColor Green
        Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    }
    # For wrapper method, use default settings
    elseif ($method -eq "wrapper") {
        $command = "python feature_selection.py --score_type $scoreType --method $method"
        Write-Host "Running: $command" -ForegroundColor Green
        Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    }
}

# Function to run model training
function Run-ModelTraining {
    param (
        [string]$scoreType,
        [string]$model,
        [string]$featureSelection
    )
    
    $fsFlag = if ($featureSelection) { "--feature_selection $featureSelection" } else { "" }
    
    # Make sure we're in the src directory
    Set-Location $srcDir
    
    $command = "python train_optimize.py --score_type $scoreType --model $model $fsFlag"
    Write-Host "Running: $command" -ForegroundColor Green
    Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    
    # Add a separator for readability
    "-" * 80 | Tee-Object -Append -FilePath $resultsFile
}

# Run all experiments with imputation
$scoreType = "FRAGIRE18"

# 1. Run feature importance calculations for all models
Write-Host "Step 1: Calculating feature importances for all models (with imputation)" -ForegroundColor Cyan
Run-FeatureImportances -scoreType $scoreType

# 2. Run for each feature selection method
foreach ($fsMethod in $featureSelectionMethods) {
    if ($fsMethod) {
        Write-Host "Step 2: Running feature selection for method: $fsMethod (with imputation)" -ForegroundColor Cyan
        Run-FeatureSelection -scoreType $scoreType -method $fsMethod
    }
    
    # 3. Train all models for this feature selection method
    Write-Host "Step 3: Training all models with feature selection: $fsMethod (with imputation)" -ForegroundColor Cyan
    foreach ($model in $models) {
        Run-ModelTraining -scoreType $scoreType -model $model -featureSelection $fsMethod
    }
}

# Return to the original directory at the end
Set-Location "p:\Thesis"

# Print completion message
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "Results saved to: $resultsFile" -ForegroundColor Green
