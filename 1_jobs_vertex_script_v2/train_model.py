import pandas as pd
from google.cloud import bigquery
import gcsfs
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sklearn
import sys
import os
from google.cloud import storage
import argparse


BUCKET_NAME = '{bucket-name}'

print("Versión de Python:", sys.version)
print("Versión de Pandas:", pd.__version__)
print("Versión de Numpy:", np.__version__)
print("Versión de Scikit-learn:", sklearn.__version__)
print("Versión de google-cloud-bigquery:", bigquery.__version__)
print("Version de gcsfs: ", gcsfs.__version__)


" ------------------- LEER ARGS PASADO AL SCRIPT ------------------- "
# Crea un objeto ArgumentParser
parser = argparse.ArgumentParser()

# Define los argumentos que esperas
parser.add_argument('--id_date_time', type=str, help='datetime ID pasado al argumento. datetime pasado de la hora que comienza a enviarse el job')

# Analiza los argumentos de la línea de comandos
args = parser.parse_args()

# Accede a los argumentos como atributos del objeto 'args'
date_time = args.id_date_time
print('El valor de date_time es:', date_time)



" ------------------- Cargar datos pkl en storage ------------------- "
## utilizar gcsfs para utilizar GCS como si fuera local
print('leer data pkl')
path_data = f'gs://{BUCKET_NAME}/poc-jobs-vertex/data.pkl'
data = pd.read_pickle(path_data)

### Separar en "x" "y" ###
x = data[['feature_1', 'feature_2', 'feature_3']]
y = data[['target']]


" ------------------- train model ------------------- "
# train model
model = LinearRegression()
model.fit(x,y)

# evaluate model
y_predicted = model.predict(x)
r2_score(y_true = y,
         y_pred = y_predicted)



" ------------------- Guardar modelo entrenado para ser registrado en vertex models - menu models ------------------- "
# DEFINIR PATH CUSTOM DONDE SE VA A GUARDAR EL MODELO. Para registrar modelo en vertex obligatoriamente debe existir el path ".../model/model.pkl"
# En los códigos posteriores se crea el path completo. Aquí se crea hasta el folder ".../model/"
# En el código que envia el job de entrenamiento se debe especificar el mismo path para decir que ahí se guardará el artefacto del modelo
path_artifact_model_vertex = f'gs://{BUCKET_NAME}/poc-jobs-vertex/modeltypeA/run_{date_time}/model/'
print('path del modelo a GCS: ', path_artifact_model_vertex)


# Save model artifact to local filesystem (doesn't persist)
artifact_filename = 'model.pkl'
local_path = artifact_filename
with open(local_path, 'wb') as model_file:
    pickle.dump(model, model_file)
print('modelo guardado local')


# Upload model artifact to Cloud Storage - try: guardar en path de GCS definido // except: error al guardar
try:
    model_directory = path_artifact_model_vertex
    storage_path = os.path.join(model_directory, artifact_filename) # generar path completo "gs//.../model/model.pkl"
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)
    print('MODELO GUARDADO EN GCS')
except Exception as e: 
    print('Error: ', str(e))
    print('MODELO NO GUARDADO EN GCS')


# delete model artifact saved locally (save locally in a job don't save permanently the file)
os.remove(artifact_filename)