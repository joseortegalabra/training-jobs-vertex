import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import sklearn
import sys

print("Versión de Python:", sys.version)
print("Versión de Pandas:", pd.__version__)
print("Versión de Numpy:", np.__version__)
print("Versión de Scikit-learn:", sklearn.__version__)


PROJECT_ID_DS = 'proyect-gcp'

### Create dataset ###
def ec(x, y, z):
    return 6*np.power(x, 3) + 5*np.power(y,2) + 10 - 7*z

np.random.seed(42)

# generate random features
len_values = 5000
x = np.random.random([len_values, 3])

# predict
y = ec(x[:, 0], x[:, 1], x[: ,2])

# add noise value y
y + np.random.random(len_values)


### train model ###
model = LinearRegression()
model.fit(x,y)

y_predicted = model.predict(x)


### evaluate model ###
r2_score(y_true = y,
         y_pred = y_predicted)



### Guardar modelo entrenado ###
print('guardar modelo entrenado')


# ## Guardar modelo entrenado para ser registrado en vertex models ###

import pickle
import os
from google.cloud import storage


#### ------------------------ REGISTRAR MODELO ENTRENADO EN VERTEX AI - MENU "MODELS" ------------------------ ####
"""
######## ESTA ES LA ÚNICA PARTE PARA TENER CUIDADO PARA QUE EL SCRIPT FUNCIONE TANTO EN UN ENTRENAMIENTO LOCAL COMO GCP ########
- Para que el modelo entrenado quede registrado en el menú "MODELS" de Vertex (artefacto del modelo obligatoriamente con el nombre "model.pkl") debe ser guardado en un bucket de GCS definido
por un path creado automático al enviar el job de entrenamiento en una variable de ambiente "AIP_MODEL_DIR" que solo existe cuando se corre el job de entrenamiento en Vertex. 

- Cuando se corre localmente la variable de ambiente no existe, por lo tanto, retorna un path: None

- Ejemplo path cloud: gs://{bucket_name}/aiplatform-custom-training-2022-05-01-13:27:34.025/model/
"""

# Save model artifact to local filesystem (doesn't persist)
artifact_filename = 'model.pkl'
local_path = artifact_filename
with open(local_path, 'wb') as model_file:
    pickle.dump(model, model_file)
print('modelo guardado local')

# Upload model artifact to Cloud Storage - try: guardar en path var de ambiente cuando se corre en cloud y registrar en vertex // except: retornar error cuando no existe path var de ambiente
print('path del modelo a GCS - AIP_MODEL_DIR: ', os.getenv("AIP_MODEL_DIR"))
try:
    model_directory = os.environ['AIP_MODEL_DIR']
    storage_path = os.path.join(model_directory, artifact_filename)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)
except Exception as e: 
    print('Error: ', str(e))
    print('MODELO NO GUARDADO EN GCS')

# delete model artifact saved locally (save locally in a job don't save permanently the file)
os.remove('model.pkl')