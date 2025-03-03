# config.py

#TRAINING_FILE_FOLD = "../input/train_folds.csv"

TRAINING_FILE = "../input/data_global.xlsx"

MODEL_OUTPUT_classification = "../models/classifiers/"
MODEL_OUTPUT_regression = "../models/regressors/"

FEATURES_IMPORTANCE_OUTPUT_classification = "../models/feature_importances/feat_imp_classification/"
FEATURES_IMPORTANCE_OUTPUT_regression = "../models/feature_importances/feat_imp_regression/"

VISUALIZATION_OUTPUT_Classification = "../visualization/classification/"
VISUALIZATION_OUTPUT_Regression = "../visualization/regression/"


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


# Chutes
COLS_TO_DROP_CHUTES = [
    "Chute_1",
    "Chute_2",
    "Chute_3",
    "Chute_4",
    "Chute_5",
    "Chute_6",
    "Nombre_de_chutes_suivi_mensuel_pendant_6_mois",
    "FROPCOM0001 SCORE",
    "FROPCOM0001",
    "Foldername"
]

# Target variables for falls classification
TARGET_CHUTE_6M = "chute_6_m"    # 0: no falls, 1: >=1 faller
TARGET_CHUTE_12M = "chute_12_m"  # 0: no falls, 1: >=2 faller
