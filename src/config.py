# config.py

import os

# Input files
TRAINING_FILE = "../input/data_global.xlsx"

# Base output directories
BASE_OUTPUT_DIR = "../output"
MODEL_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "models")
VISUALIZATION_BASE_DIR = os.path.join(BASE_OUTPUT_DIR, "visualization")

# Create a function to generate paths with processing steps
def get_output_paths(imputation=True, feature_selection=None):
    """
    Generate output paths based on processing steps
    
    Parameters:
    -----------
    imputation : bool
        Whether imputation is used
    feature_selection : str or None
        Feature selection method: 'wrapper', 'embedded', or None
        
    Returns:
    --------
    dict
        Dictionary with output paths
    """
    # Create processing steps string
    steps = []
    if imputation:
        steps.append("imputed")
    else:
        steps.append("raw")
        
    if feature_selection:
        steps.append(f"{feature_selection}_fs")
    
    # Join steps with underscore
    step_str = "_".join(steps)
    
    # Create paths
    paths = {
        "models": os.path.join(MODEL_BASE_DIR, step_str, "classifiers"),
        "feature_importances": os.path.join(MODEL_BASE_DIR, step_str, "feature_importances"),
        "visualization": os.path.join(VISUALIZATION_BASE_DIR, step_str, "classification")
    }
    
    return paths

# Default paths for backward compatibility
MODEL_OUTPUT = os.path.join(MODEL_BASE_DIR, "imputed", "classifiers")
FEATURE_IMPORTANCE_DIR = os.path.join(MODEL_BASE_DIR, "imputed", "feature_importances")
VISUALIZATION_OUTPUT = os.path.join(VISUALIZATION_BASE_DIR, "imputed", "classification")

# Columns to drop
COLS_TO_DROP_FRAGIRE18_FRIED = [
    "Foldername",
    "Frailty_State_GFST",
    "Fried_State",
    "Frailty_Score_FRAGIRE18_SQ001",
    "Fried_Score_FRIED_TOTAL_Version_1",
    "FRIED - ITEM PERTE DE POIDS -  mesuré",
    "FRIED - ITEM PERTE DE POIDS - déclaré MNA",
    "FRIED - ITEM - FATIGUE SF36-Q29",
    "FRIED - ITEM FATIGUE SOUS SCORE FRIED ENERGY-FATIGUE",
    "FRIED - ITEM - Activité Physique",
    "FRIED ITEM - Walk time_4m et temps SPPB",
    "FRIED ITEM - Walk time - GAIT",
    "FRIED ITEM GRIP",
    "FRIED_Version_3",
    "has_followup_romberg", 
    "has_followup_gait", 
    "has_followup_grip",
]

# Drop folow-up columns
COLS_TO_DROP_FOLLOW_UP_PATTERN = "*_follow-up"

# Dictionary mapping score types to their target columns
TARGET_COLUMNS_DICT = {
    'FRIED': {
        'classification': 'Fried_State'
    },
    'FRAGIRE18': {
        'classification': 'Frailty_State_GFST'
    }
}
