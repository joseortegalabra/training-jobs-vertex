import pandas as pd
from google.cloud import bigquery
import gcsfs
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import sklearn
import sys
import os
from google.cloud import storage
import argparse
from google.cloud import aiplatform as vertex_ai
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

print("Versión de Python:", sys.version)
print("Versión de Pandas:", pd.__version__)
print("Versión de Numpy:", np.__version__)
print("Versión de Scikit-learn:", sklearn.__version__)
print("Versión de google-cloud-bigquery:", bigquery.__version__)
print("Version de gcsfs: ", gcsfs.__version__)




" ------------------- AUXILIAR FUNCTIONS ------------------- "
def evaluate_model(y_true, y_predicted):
    """
    Given "y_true" and "y_predicted" calculate metrics of performance (r2, rmse, mae)
    """
    r2_metric = r2_score(y_true, y_predicted)

    rmse_metric = mean_squared_error(y_true, y_predicted, squared = False)

    mae_metric = mean_absolute_error(y_true, y_predicted)

    print("r2: ", r2_metric)
    print("rmse: ", rmse_metric)
    print("mae_metric: ", mae_metric)
    return r2_metric, rmse_metric, mae_metric

def plot_y_true_vs_y_pred(y, y_pred, title_plot):
    """
    Plot y_true vs y_pred (using matplotlib figure). y_true in X-axis, y_pred in Y-axis.

    Args:
        y (dataframe): dataframe with y-true values 
        y_pred (dataframe): dataframe with y-pred values
        title_plot (string): tittle in the plot
    
    Return
        fig (figure matplolib): figure to show, download, etc
    """
    fig, ax = plt.subplots()
    scatter_plot = ax.scatter(y, y_pred, alpha=0.3, marker='x', label='y_true vs y_pred')

    # Add bisectriz
    y_bisectriz = x_bisectriz = np.linspace(y.min()[0], y.max()[0], y.shape[0])
    ax.plot(x_bisectriz, y_bisectriz, label='Bisectriz', color='red', alpha=0.3)

    # Add names to axis
    ax.set_xlabel('Y true')
    ax.set_ylabel('Y pred')
    
    ax.set_title(title_plot)
    ax.legend()


    # save fig, return the local path and close fig
    name_y_true_y_pred = 'y_true_y_pred.png'
    plt.savefig(name_y_true_y_pred)
    plt.close()
    
    return fig, name_y_true_y_pred

def create_instance_tensorboard(experiment_name, experiment_description, PROJECT_ID_DS, location_gcp):
    """
    Create a vertex tensorboard instance. The instance of tensorboard is created with the idea to have the same name of the experiment of vertex ai
    that will use this instance of vertex tensorboard.

    Obs: This code create always a tensorboard instance, with the same name (display_name) but different ID, so it is necessary RUN ONCE
    
    Args
        experiment_name (string)
        experiment_description (string)
        PROJECT_ID_DS (string)
        location_gcp (string)

    Return
        id_experiment_tensorboard (vertex ai tensorboard object)
    """
    id_tensorboard_vertex = vertex_ai.Tensorboard.create(display_name = f'tensorboard-{experiment_name}',
                                                          description = f'tensorboard-{experiment_description}',
                                                          project = PROJECT_ID_DS,
                                                          location = location_gcp
                                                         )
    return id_tensorboard_vertex

def get_tensorboard_instance_or_create(experiment_name, experiment_description, PROJECT_ID_DS, location_gcp):
    """
    Search if exist a tensorboard instance and get it. If the instance doesn't exist, create it.
    The instance of tensorboard has its name with the idea to have the same name of the experiment of vertex ai that will use this instance
    of vertex.

    Args
        experiment_name (string)
        experiment_description (string)
        PROJECT_ID_DS (string)
        location_gcp (string)

    Return
        id_experiment_tensorboard (vertex ai tensorboard object)
    """
    
    ''' search tensorboard instance. if the list is empty the tensorboard instance doesn't exist and it will created '''
    # GET tensorboard instance created FILTERING by display name. return a list of the instance doesn't exist return a empty list
    list_tensorboard_vertex = vertex_ai.Tensorboard.list(
        filter = f'display_name="tensorboard-{experiment_name}"',
        project = PROJECT_ID_DS,
        location = location_gcp
    )

    # if vertex tensorboard instance doesn't exist, create it
    if len(list_tensorboard_vertex) == 0:
        print('--- creating vertex tensorboard instance ---')
        id_tensorboard_vertex = vertex_ai.Tensorboard.create(display_name = f'tensorboard-{experiment_name}',
                                                                 description = f'tensorboard-{experiment_description}',
                                                                 project = PROJECT_ID_DS,
                                                                 location = location_gcp
                                                                ) # return tensorboard instance created
    else:
        print('--- tensorboard instance already exists ---')
        id_tensorboard_vertex = list_tensorboard_vertex[0] # tensorboard instance exists, return it
    
    return id_tensorboard_vertex

def save_local_to_gcs(uri_gcs, uri_local):
    """
    AUXILIAR. Save a locally file onto GCS.
    Args:
        uri_gcs (string): path in gcs where the local file will be saved
        uri_local (strring). path in local where the local file was saved

    Return
        nothing
    """

    blob = storage.blob.Blob.from_string(uri_gcs, client=storage.Client())
    blob.upload_from_filename(uri_local)

def save_artifacts_experiments_vertex(path_artifact_locally, type_artifact, bucket_gcs, experiment_name, run_name):
    """
    Save an artifact in experiments in vertex. This functions works for an individual artifact. The run of the experiment needs to be created
    The input is a file saved locally and the output is the file registered as a artifact of a run of a vertex experiment
    
    There following steps are necesarys to save the artifact
    - save artifact locally
    - save artifact in GCS
    - link the artifact in GCS with vertex metadata
    - link vertex metadata with an artifact saved in a run (experiment vertex)
    - delete the file locally
    """

    # 1. save artifact locally (done -input function)


    # 2. save artifact in GCS
    path_artifact_gcs = f'gs://{bucket_gcs}/{experiment_name}/{run_name}/{path_artifact_locally}'
    save_local_to_gcs(uri_gcs = path_artifact_gcs, 
                      uri_local = path_artifact_locally)

    
    # 3. link the artifact in GCS with vertex metadata
    path_artifact_locally_corrected = path_artifact_locally.replace('_', '-').replace('.', '-') # in the name only accepted "-"
    path_artifact_locally_corrected = path_artifact_locally_corrected.lower() # in the name only acceted lower case [a-z0-9][a-z0-9-]{0,127}
    
    
    artifact_metadata = vertex_ai.Artifact.create(
        schema_title = "system.Artifact", 
        uri = path_artifact_gcs, # 
        display_name = f"artifact-{path_artifact_locally}", # nombre con el que se muestra en el menu "metadata"
        description = f"description-{path_artifact_locally}",
        resource_id = f"{path_artifact_locally_corrected}-{experiment_name}-{run_name}"  # nombre con el que se muestra en el menu "artifact del run del experimento" de vertex. No acepta espacios
        )


    # 4. link vertex metadata with an artifact saved in a run 
    executions = vertex_ai.start_execution(
        schema_title="system.ContainerExecution", 
        display_name='REGISTRO DE ARTIFACTS'
    )
    executions.assign_input_artifacts([artifact_metadata])

    
    # 5. delete the file local
    #os.remove(path_artifact_locally)




" ------------------- PARAMS ------------------- "
PROJECT_ID_DS = '{PROJECT-GCP}' # project gcp
BUCKET_NAME = '{bucket-name}' # bucket artifacts
REGION = '{region}' # region
EXPERIMENT_NAME_REGISTRY = 'jobs-registry-vertex' # name of the experiment in "VERTEX EXPERIMENT" where the run of trainning model was saved



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
X = data[['feature_1', 'feature_2', 'feature_3']]
y = data[['target']]



" ------------------- INITIALIZE VERTEX EXPERIMENTS  ------------------- "
# PARAMS
EXPERIMENT_NAME = EXPERIMENT_NAME_REGISTRY # the name of the vertex experiment is the name of the dataset
EXPERIMENT_DESCRIPTION = f'Develop job vertex with registry in vertex experiment. Run forecasting models of a target'

# search tensorboard instance, if it doesn't exist -> created it
id_tensorboard_vertex = get_tensorboard_instance_or_create(experiment_name = EXPERIMENT_NAME,
                                                           experiment_description = EXPERIMENT_DESCRIPTION,
                                                           PROJECT_ID_DS = PROJECT_ID_DS,
                                                           location_gcp = REGION
                                                          )

# set experiment (or created if it doesn't exist - automatically)
print('\n--- setting experiment vertex ai ---')
vertex_ai.init(
    experiment = EXPERIMENT_NAME,
    experiment_description = EXPERIMENT_DESCRIPTION,
    experiment_tensorboard = id_tensorboard_vertex,
    project = PROJECT_ID_DS,
    location = REGION,
    )



" ------------------- TRAIN MODEL AND REGISRY IN VERTEX EXPERIMENTS ------------------- "

""" RUN NAME IN EXPERIMENT """
RUN_NAME = "run-test-job2"
print('---- trainning model: ', RUN_NAME)


""" train model """
# define params to save. In a dicctionary
params_training = {
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 10,
    'random_state': 42
}

# create model - train it - evaluate it
tree = DecisionTreeRegressor(**params_training) # create model
tree.fit(X, y) # train
y_predicted = tree.predict(X) # predict
r2_tree, rmse_tree, mae_tree = evaluate_model(y, y_predicted) # evaluate metrics
plot_y_true_y_pred, path_y_true_y_pred = plot_y_true_vs_y_pred(y = y, y_pred = y_predicted, title_plot = f'model: {RUN_NAME}') # Ytrue_vs_Ypred


""" registry run in experiment """
# create a run
vertex_ai.start_run(RUN_NAME)

# parameters of the model trained
vertex_ai.log_params(params_training)

# define metrics to save. In a dicctionary
metrics_to_save = {
    'r2': r2_tree,
    'rmse': rmse_tree,
    'mae': mae_tree
}

# save metrics
vertex_ai.log_metrics(metrics_to_save)

# save graphs
print('saving plot y_true vs y_pred ...')
save_artifacts_experiments_vertex(path_artifact_locally = path_y_true_y_pred, # plot y_true vs y_pred
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_NAME, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save model (but not registry)
print('saving model ...')
model_name = 'model.pkl'
with open(model_name, "wb") as output: # save locally
    pickle.dump(tree, output)
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = model_name,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_NAME, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save X
print('saving X ...')
artifact_data = 'X.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(X, output)# change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_NAME, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

# save y
print('saving y ...')
artifact_data = 'y.pkl' # change path
with open(artifact_data, "wb") as output: 
    pickle.dump(y, output) # change python variable with artifact
    output.close()
save_artifacts_experiments_vertex(path_artifact_locally = artifact_data,
                                  type_artifact = 'artifact', 
                                  bucket_gcs = BUCKET_NAME, 
                                  experiment_name = EXPERIMENT_NAME, 
                                  run_name = RUN_NAME
                                 )

### terminar run
vertex_ai.end_run()


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
    pickle.dump(tree, model_file)
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