from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def train_lr_model(data):
    """
    Dado un dataframe entrenar una regresión lineal
    Args:
        data (dataframe)
    
    Return
        model (model sklearn)
    """

    " ------------------- PREPROCESS DATA ------------------- "
    ### Separar "data" en "x" "y" ###
    x = data[['feature_1', 'feature_2', 'feature_3']]
    y = data[['target']]


    " ------------------- TRAIN MODEL ------------------- "
    # train model
    model = LinearRegression()
    model.fit(x,y)

    # evaluate model
    y_predicted = model.predict(x)
    r2_score(y_true = y,
            y_pred = y_predicted)
    print('R2_SCORE: ', r2_score)


    # return model
    return model