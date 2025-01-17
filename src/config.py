# config.py

#TRAINING_FILE_FOLD = "../input/train_folds.csv"

TRAINING_FILE = "../input/data_global.xlsx"

MODEL_OUTPUT = "../models/classifiers/"

FEATURES_IMPORTANCE_OUTPUT = "../models/feature_importances/"

VISUALIZATION_OUTPUT = "../visualization/svg/"


COLS_TO_DROP_FRAGIRE18_FRIED = [
    "Foldername",
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