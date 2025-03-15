# config.py

TRAINING_FILE = "../input/data_global.xlsx"

MODEL_OUTPUT = "../models/classifiers/"
FEATURE_IMPORTANCE_DIR = "../models/feature_importances/"
VISUALIZATION_OUTPUT = "../visualization/classification/"

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
