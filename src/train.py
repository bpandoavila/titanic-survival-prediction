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

# Entrenar modelo

#data = read_file_csv(data_processed_folder, "titanic_train.csv")
data = read_file_csv(data_processed_folder, "titanic_train.csv", index_col='PassengerId')

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
print('Modelo entrenado')

# Guardando el modelo entrenado
package = os.path.join(data_model_folder, 'best_model')
print(package)
dump(model, package)

#pickle.dump(model, open(package, 'wb'))
print('Modelo exportado correctamente en la carpeta models')
