from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#import pickle
from joblib import dump, load

import os

# Librerias

import pandas as pd

def read_file_csv(data_folder, filename, index_col=None):
    path = os.path.join(data_folder, filename)  # Ruta absoluta
    df = pd.read_csv(path)
    if index_col:
        df.set_index(index_col, inplace=True)
    print(f"{filename} cargado. Dimensiones: {df.shape}")
    display(df.head())
    return df


# Scoring del Modelo

data = read_file_csv(data_processed_folder, "titanic_score.csv", index_col='PassengerId')

# Leemos el modelo entrenado para usarlo
package = os.path.join(data_model_folder, 'best_model')
model = load(package)
print('Modelo importado correctamente')

# Predecimos sobre el set de datos de Scoring
scores = 'final_score.csv'    
res = model.predict(data).reshape(-1,1)
pred = pd.DataFrame(res, columns=['PREDICT'])
ruta_file_scoring = os.path.join(data_scores_folder, scores)
print(ruta_file_scoring)
pred.to_csv(ruta_file_scoring)
print(scores, 'exportado correctamente en la carpeta scores')

