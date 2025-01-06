from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

def encode_profession_columns(df, columns=['statut', 'profession', 'niveau_etudes']):
    """
    Encode les colonnes de profession pour le machine learning
    
    Paramètres:
    df : DataFrame contenant les colonnes à encoder
    columns : liste des colonnes à encoder (par défaut: ['statut', 'profession', 'niveau_etudes'])
    
    Retourne:
    DataFrame encodé et dictionnaires des encodeurs pour chaque colonne
    """
    encoded_df = df.copy()
    encoders = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Remplacer les valeurs manquantes par une catégorie spéciale
        encoded_df[col] = encoded_df[col].fillna('MISSING')
        
        if col == 'statut':
            # Pour le statut, utiliser LabelEncoder car c'est une variable catégorielle simple
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col])
            encoders[col] = {'type': 'label', 'encoder': le}
            
        elif col == 'profession':
            # Pour la profession, utiliser OneHotEncoder car il y a beaucoup de catégories
            # et pas forcément d'ordre entre elles
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            profession_encoded = ohe.fit_transform(encoded_df[[col]])
            
            # Créer les nouvelles colonnes avec les professions encodées
            profession_cols = [f'profession_{i}' for i in range(profession_encoded.shape[1])]
            profession_df = pd.DataFrame(profession_encoded, columns=profession_cols)
            
            # Supprimer l'ancienne colonne et ajouter les nouvelles
            encoded_df = pd.concat([encoded_df.drop(col, axis=1), profession_df], axis=1)
            encoders[col] = {'type': 'onehot', 'encoder': ohe, 'columns': profession_cols}
            
        elif col == 'niveau_etudes':
            # Pour le niveau d'études, créer un mapping ordinal basé sur le niveau
            niveau_mapping = {
                'MISSING': 0,
                'ANALPHABETE': 1,
                'ILLETRE': 1,
                'NE SAIT PAS LIRE': 1,
                'CP': 2,
                'CE1': 3,
                'CE2': 4,
                'CM1': 5,
                'CM2': 6,
                'SIXIEME': 7,
                'CINQUIEME': 8,
                'QUATRIEME': 9,
                'TROISIEME': 10,
                'SECONDE': 11,
                'PREMIERE': 12,
                'TERMINALE': 13,
                'BAC': 14,
                'BAC +2': 15,
                'BAC +3': 16,
                'MASTER': 17,
                'BAC +7': 18
            }
            
            # Fonction pour trouver le niveau le plus proche dans le mapping
            def find_closest_level(x):
                if pd.isna(x):
                    return niveau_mapping['MISSING']
                x = str(x).upper()
                # Chercher une correspondance exacte d'abord
                for key in niveau_mapping:
                    if key in x:
                        return niveau_mapping[key]
                return niveau_mapping['MISSING']
            
            encoded_df[col] = encoded_df[col].apply(find_closest_level)
            encoders[col] = {'type': 'ordinal', 'mapping': niveau_mapping}
    
    return encoded_df, encoders

def transform_new_data(df, encoders, columns=['statut', 'profession', 'niveau_etudes']):
    """
    Transforme de nouvelles données en utilisant les encodeurs existants
    """
    transformed_df = df.copy()
    
    for col in columns:
        if col not in df.columns or col not in encoders:
            continue
            
        transformed_df[col] = transformed_df[col].fillna('MISSING')
        
        if encoders[col]['type'] == 'label':
            transformed_df[col] = encoders[col]['encoder'].transform(transformed_df[col])
            
        elif encoders[col]['type'] == 'onehot':
            transformed = encoders[col]['encoder'].transform(transformed_df[[col]])
            transformed_df = pd.concat([
                transformed_df.drop(col, axis=1),
                pd.DataFrame(transformed, columns=encoders[col]['columns'])
            ], axis=1)
            
        elif encoders[col]['type'] == 'ordinal':
            transformed_df[col] = transformed_df[col].apply(
                lambda x: encoders[col]['mapping'].get(x, encoders[col]['mapping']['MISSING'])
            )
    
    return transformed_df

# Exemple d'utilisation:
"""
# Pour encoder les données d'entraînement :
encoded_df, encoders = encode_profession_columns(df)

# Pour transformer de nouvelles données avec les mêmes encodeurs :
new_encoded_df = transform_new_data(new_df, encoders)
"""
