# import scripts
import trainer.generate_data as gen_data
import trainer.train_model as train

import pandas as pd
import numpy as np
import sklearn
import sys
from google.cloud import bigquery
import gcsfs
import argparse
import pickle
from google.cloud import storage
import os

print("Versión de Python:", sys.version)
print("Versión de Pandas:", pd.__version__)
print("Versión de Numpy:", np.__version__)
print("Versión de Scikit-learn:", sklearn.__version__)
print("Versión de google-cloud-bigquery:", bigquery.__version__)
print("Version de gcsfs: ", gcsfs.__version__)


" ------------------- PARÁMETROS ------------------- "
PROJECT_ID_DS = '{project-gcp}'
BUCKET_NAME = '{bucket-name}'



" ------------------- LEER ARGS PASADO AL SCRIPT ------------------- "
# get args
parser = argparse.ArgumentParser()
parser.add_argument('--id_date_time', type=str, help='datetime ID pasado al argumento. datetime pasado de la hora que comienza a enviarse el job')
args = parser.parse_args()

# assign args
date_time = args.id_date_time
print('El valor de date_time es:', date_time)



" ------------------- GENERATE DATA. get dataframe with data ------------------- "
len_data = 70000
data = gen_data.generate_data(len_data)



" ------------------- TRAIN MODEL. get an artifact model trained ------------------- "
model = train.train_lr_model(data)




" ------------------- SAVE MODEL. save pkl model in a custom gcs path ------------------- "

# DEFINIR PATH CUSTOM DONDE SE VA A GUARDAR EL MODELO. Para registrar modelo en vertex obligatoriamente debe existir el path ".../model/model.pkl"
# En los códigos posteriores se crea el path completo. Aquí se crea hasta el folder ".../model/"
path_artifact_model_vertex = f'gs://{BUCKET_NAME}/poc-jobs-vertex/modeltypeB/run_{date_time}/model/'
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