from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def encode_profession_physical(df, profession_column='profession'):
    """
    Encode les professions selon leur niveau d'effort physique
    
    Catégories :
    0 : Non spécifié/Inconnu
    1 : Effort physique faible (ex: travail de bureau, enseignement)
    2 : Effort physique modéré (ex: commerce, artisanat)
    3 : Effort physique élevé (ex: agriculture, BTP, industrie)
    """
    
    # Mapping des professions par niveau d'effort physique
    effort_mapping = {
        # Effort physique élevé (3)
        'AGRICOLE': 3,
        'AGRICULTEUR': 3,
        'AGRICULTURE': 3,
        'BTP': 3,
        'BATIMENT': 3,
        'MARINE': 3,
        'ESPACES VERTS': 3,
        'OUVRIER': 3,
        'OUVRIERE': 3,
        'SAISONNIER': 3,
        'CARRELEUR': 3,
        'INDUSTRIE': 3,
        
        # Effort physique modéré (2)
        'ARTISAN': 2,
        'ARTISANAT': 2,
        'COMMERCE': 2,
        'RESTAURATION': 2,
        'CANTINIERE': 2,
        'AIDE SOIGNANT': 2,
        'AIDE SOIGNANTE': 2,
        'AUXILIAIRE DE VIE': 2,
        'AUXILIAIRE PUERICULTURE': 2,
        'ASSISTANTE MATERNELLE': 2,
        
        # Effort physique faible (1)
        'ENSEIGNANT': 1,
        'EDUCATION NATIONALE': 1,
        'ADMINISTRATIF': 1,
        'POSTE': 1,
        'FONCTION PUBLIQUE': 1,
        'GESTIONNAIRE': 1,
        'COMPTABILITE': 1,
        'GRAPHISTE': 1,
        'FORMATEUR': 1,
        'EXPERT': 1
    }
    
    def categorize_physical_effort(profession):
        if pd.isna(profession):
            return 0
        
        profession = str(profession).upper()
        
        # Chercher les mots-clés dans la description de la profession
        for key, value in effort_mapping.items():
            if key in profession:
                return value
                
        return 0  # Catégorie par défaut si non trouvé
    
    encoded_df = df.copy()
    
    # Créer la nouvelle colonne avec le niveau d'effort physique
    encoded_df['niveau_effort_physique'] = encoded_df[profession_column].apply(categorize_physical_effort)
    
    # Créer un encodeur pour garder la trace du mapping
    encoders = {
        'type': 'ordinal',
        'mapping': {
            0: 'Non spécifié/Inconnu',
            1: 'Effort physique faible',
            2: 'Effort physique modéré',
            3: 'Effort physique élevé'
        }
    }
    
    return encoded_df, encoders

def transform_new_profession_physical(df, profession_column='profession'):
    """
    Applique le même encodage à de nouvelles données
    """
    return encode_profession_physical(df, profession_column)

# Exemple d'utilisation :
"""
# Pour encoder les données
encoded_df, encoders = encode_profession_physical(df, 'profession')

# Pour voir la distribution des niveaux d'effort
print(encoded_df['niveau_effort_physique'].value_counts())

# Pour voir le mapping utilisé
print(encoders['mapping'])
"""
