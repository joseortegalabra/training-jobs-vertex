El más básico V2


- LEER DATOS GUARDADOS EN UN PKL EN GCS, leer datos pkl de GCS, entrenar modelo y guardarlo para ser registrado en models de vertex
(para hacer esto basta con instalar más packages en la docker image base de gcp de entrenamiento y que sean compatibles con las versiones)


- Cambiar path por defecto donde se guarda el pkl obtenido del entrenamiento (AIP_MODEL_DIR) por un path personalizado custom

- Guardar script de entrenamiento .tar.gz en un bucket destinado a guardar esos artefacto, mientras que el artefacto del modelo se guarda en el
path personalizado custom en su propio bucket (pero ambos bucket en el mismo proyecto de GCP por temas de permisos)

- Registrar modelo en servicio de vertex guardado en un GCS personalizado

- Enviar args al script .py de entrenamiento
