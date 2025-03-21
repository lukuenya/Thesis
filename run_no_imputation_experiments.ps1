# Define path to the results file
$resultsFile = "p:\Thesis\no_imputation_results.txt"
$srcDir = "p:\Thesis\src"

# Clear or create the results file
"Results for no-imputation experiments" | Out-File -FilePath $resultsFile -Force

# Define models and feature selection methods
$models = @("lightgbm", "xgboost", "catboost", "randomforest")
$featureSelectionMethods = @("embedded", "wrapper", $null)

# Function to run feature importances for all models
function Run-FeatureImportances {
    param (
        [string]$scoreType,
        [switch]$noImputation
    )
    
    $impFlag = if ($noImputation) { "--no_imputation" } else { "" }
    
    foreach ($model in $models) {
        # Make sure we're in the src directory
        Set-Location $srcDir
        
        $command = "python get_feature_importances.py --score_type $scoreType --model_name $model $impFlag"
        Write-Host "Running: $command" -ForegroundColor Green
        Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    }
}

# Function to run feature selection
function Run-FeatureSelection {
    param (
        [string]$scoreType,
        [string]$method,
        [switch]$noImputation
    )
    
    $impFlag = if ($noImputation) { "--no_imputation" } else { "" }
    
    # Make sure we're in the src directory
    Set-Location $srcDir
    
    # For embedded method, use threshold 10
    if ($method -eq "embedded") {
        $command = "python feature_selection.py --score_type $scoreType --method $method --threshold 10 $impFlag"
        Write-Host "Running: $command" -ForegroundColor Green
        Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    }
    # For wrapper method, use default settings
    elseif ($method -eq "wrapper") {
        $command = "python feature_selection.py --score_type $scoreType --method $method $impFlag"
        Write-Host "Running: $command" -ForegroundColor Green
        Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    }
}

# Function to run model training
function Run-ModelTraining {
    param (
        [string]$scoreType,
        [string]$model,
        [string]$featureSelection,
        [switch]$noImputation
    )
    
    $impFlag = if ($noImputation) { "--no_imputation" } else { "" }
    $fsFlag = if ($featureSelection) { "--feature_selection $featureSelection" } else { "" }
    
    # Make sure we're in the src directory
    Set-Location $srcDir
    
    $command = "python train_optimize.py --score_type $scoreType --model $model $fsFlag $impFlag"
    Write-Host "Running: $command" -ForegroundColor Green
    Invoke-Expression $command | Tee-Object -Append -FilePath $resultsFile
    
    # Add a separator for readability
    "-" * 80 | Tee-Object -Append -FilePath $resultsFile
}

# Run all experiments with --no_imputation
$scoreType = "FRIED"

# 1. Run feature importance calculations for all models with --no_imputation
Write-Host "Step 1: Calculating feature importances for all models (no imputation)" -ForegroundColor Cyan
Run-FeatureImportances -scoreType $scoreType -noImputation

# 2. Run for each feature selection method
foreach ($fsMethod in $featureSelectionMethods) {
    if ($fsMethod) {
        Write-Host "Step 2: Running feature selection for method: $fsMethod (no imputation)" -ForegroundColor Cyan
        Run-FeatureSelection -scoreType $scoreType -method $fsMethod -noImputation
    }
    
    # 3. Train all models for this feature selection method
    Write-Host "Step 3: Training all models with feature selection: $fsMethod (no imputation)" -ForegroundColor Cyan
    foreach ($model in $models) {
        Run-ModelTraining -scoreType $scoreType -model $model -featureSelection $fsMethod -noImputation
    }
}

# Return to the original directory at the end
Set-Location "p:\Thesis"

# Print completion message
Write-Host "All experiments completed!" -ForegroundColor Green
Write-Host "Results saved to: $resultsFile" -ForegroundColor Green
