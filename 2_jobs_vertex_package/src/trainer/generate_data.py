import pandas as pd
import numpy as np

def ec(x, y, z):
    return 6*np.power(x, 3) + 5*np.power(y,2) + 10 - 7*z


def generate_data(len_values):
    """
    Generar data para ejemplo.
    Args:
        len_values (int): cantidad de datos a generarse

    Return:
        data (dataframe): dataframe de tamaño (len_values, 3)
    """

    # seed
    np.random.seed(42)

    # generate random features
    x = np.random.random([len_values, 3])

    # predict
    y = ec(x[:, 0], x[:, 1], x[: ,2])

    # add noise value y
    y = y + np.random.random(len_values)

    # transform into a dataframe
    data = pd.DataFrame(x, columns = ['feature_1', 'feature_2', 'feature_3'])
    data['target'] = y

    # return dataframe
    return data