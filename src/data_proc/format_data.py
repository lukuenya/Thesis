import pandas as pd
import numpy as np

def format_profession_education(data):
    def split_info(text):
        if pd.isna(text):
            return pd.Series({'statut': np.nan, 'profession': np.nan, 'niveau_etudes': np.nan})
        
        # Convertir en majuscules pour uniformisation
        text = str(text).upper()
        
        # Séparation par "/"
        parts = [part.strip() for part in text.split('/')]
        
        # Initialisation des variables
        statut = ''
        profession = ''
        niveau_etudes = ''
        
        # Si une seule partie
        if len(parts) == 1:
            if 'RETRAIT' in parts[0]:
                if 'ANALPHABETE' in parts[0]:
                    profession = parts[0].replace('ANALPHABETE', '').strip()
                    niveau_etudes = 'ANALPHABETE'
                else:
                    profession = parts[0]
            else:
                profession = parts[0]
        
        # Si deux parties ou plus
        else:
            profession = parts[0]
            niveau_etudes = parts[1]
        
        # Extraction du statut
        if 'RETRAIT' in profession:
            statut = 'RETRAITÉ(E)'
            profession = profession.replace('RETRAITE', '').replace('RETRAITEE', '').strip()
        elif 'RSA' in profession:
            statut = 'RSA'
            profession = profession.replace('RSA', '').strip()
        else:
            statut = 'AUTRE'
        
        # Nettoyage
        profession = profession.strip()
        if profession.startswith('E '):
            profession = profession[2:]
        
        return pd.Series({
            'statut': statut,
            'profession': profession if profession else np.nan,
            'niveau_etudes': niveau_etudes if niveau_etudes else np.nan
        })

    # Appliquer la fonction à chaque ligne
    result = pd.DataFrame([split_info(x) for x in data])
    return result

# Exemple d'utilisation
# df['nouvelle_colonne'] = format_profession_education(df['colonne_originale'])
