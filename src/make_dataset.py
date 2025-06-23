# --- Configuración inicial ---
#!pip install pandas scikit-learn matplotlib

# Creación y definición de Carpetas

import os

data_raw_folder = '/content/data/raw'
data_processed_folder = '/content/data/processed'
data_model_folder = '/content/models'
data_scores_folder = '/content/data/scores'

os.makedirs(data_raw_folder, exist_ok=True)
os.makedirs(data_processed_folder, exist_ok=True)
os.makedirs(data_model_folder, exist_ok=True)
os.makedirs(data_scores_folder, exist_ok=True)

train_url = os.path.join(data_raw_folder, "train.csv")
print(train_url)

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

def data_preparation(df):
    # Eliminar columnas no útiles
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    display(df.head())
    # Rellenar valores faltantes
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Sin inplace
    display(df.head())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    display(df.head())
    # Convertir variables categóricas
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    display(df.head())
    return df

def data_exporting(df, filename):
    features = df.columns.tolist()
    dfp = df[features]
    dfp.to_csv(os.path.join(data_processed_folder, filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Preparación de Datos

df1 = read_file_csv(data_raw_folder, "train.csv", index_col='PassengerId')
tdf1 = data_preparation(df1)
data_exporting(tdf1, 'titanic_train.csv')

df2 = read_file_csv(data_raw_folder, "test.csv", index_col='PassengerId')
tdf2 = data_preparation(df2)
data_exporting(tdf2, 'titanic_score.csv')

