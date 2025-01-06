# config.py

TRAINING_FILE_FOLD = "../input/train_folds.csv"

TRAINING_FILE = "../input/df_merged_v2.xlsx"

MODEL_OUTPUT = "../models/"


COLS_TO_DROP_FRAILTY_SCORE = [
    "Frailty_Score_FRAGIRE18_SQ001",
    "Fried_Score_FRIED_TOTAL_Version_1",
    "FRIED - ITEM PERTE DE POIDS -  mesuré",
    "FRIED - ITEM PERTE DE POIDS - déclaré MNA",
    "FRIED - ITEM - FATIGUE SF36-Q29",
    "FRIED - ITEM FATIGUE SOUS SCORE FRIED ENERGY-FATIGUE",
    "FRIED - ITEM - Activité Physique",
    "FRIED ITEM - Walk time (4m et temps SPPB)",
    "FRIED ITEM - Walk time - GAIT",
    "FRIED ITEM GRIP",
    "FRIED Version 3 (perte de poids mesuré et vitesse de marche avec GAIT)"
    ]

COLS_TO_DROP_FRIED_SCORE = [
    "Fried_Score_FRIED_TOTAL_Version_1",
    "FRIED - ITEM PERTE DE POIDS -  mesuré",
    "FRIED - ITEM PERTE DE POIDS - déclaré MNA",
    "FRIED - ITEM - FATIGUE SF36-Q29",
    "FRIED - ITEM FATIGUE SOUS SCORE FRIED ENERGY-FATIGUE",
    "FRIED - ITEM - Activité Physique",
    "FRIED ITEM - Walk time (4m et temps SPPB)",
    "FRIED ITEM - Walk time - GAIT",
    "FRIED ITEM GRIP",
    "FRIED Version 3 (perte de poids mesuré et vitesse de marche avec GAIT)"
    ]

