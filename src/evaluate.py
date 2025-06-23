from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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

# Evaluar Modelo

data = read_file_csv(data_processed_folder, "titanic_train.csv", index_col='PassengerId')
X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Leemos el modelo entrenado para usarlo
package = os.path.join(data_model_folder, 'best_model')
model = load(package)
print('Modelo importado correctamente')

# Predecimos sobre el set de datos de validación 
y_pred_test=model.predict(X_test)

# Generamos métricas de diagnóstico
cm_test = confusion_matrix(y_test,y_pred_test)
print("Matriz de confusion: ")
print(cm_test)
accuracy_test=accuracy_score(y_test,y_pred_test)
print("Accuracy: ", accuracy_test)
precision_test=precision_score(y_test,y_pred_test)
print("Precision: ", precision_test)
recall_test=recall_score(y_test,y_pred_test)
print("Recall: ", recall_test)
print('Finalizó la validación del Modelo')
