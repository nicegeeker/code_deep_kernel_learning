import numpy as np
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

import properscoring as ps

def k_nearest_neighbor(train_X, train_y, coords, k, weights="uniform"):
    """[]

    Args:
        train_X ([numpy.Arrary]): [shape(N, 2), location of training points.]
        train_y ([numpy.Array]): [shape(N)]
        coords ([numpy.Array]): [shape(res*res, 2), coords for prediction.]
        k ([int]): [Number of neighbors.]
        weights (str, optional): ['uniform' or 'distance']. Defaults to 'uniform'.

    Returns:
        [type]: [description]
    """
    knn = KNeighborsRegressor(k, weights=weights)
    predict = knn.fit(train_X, train_y).predict(coords)
    return predict

def ok(train_X, train_y, coords):
    ok = OrdinaryKriging(train_X[:, 0], 
                         train_X[:, 1], train_y, variogram_model="exponential")
    predict_mean, predict_var = ok.execute("points", coords[:, 0], coords[:, 1])
    return predict_mean, predict_var

def uk(train_X, train_y, coords):
    uk = UniversalKriging(train_X[:, 0], 
                         train_X[:, 1], train_y, variogram_model="exponential")
    predict_mean, predict_var = uk.execute("points", coords[:, 0], coords[:, 1])
    return predict_mean, predict_var


def MAPE(true_rss, predict, training_idx=None):
    if training_idx is None:
        return mean_absolute_percentage_error(true_rss, predict)
    else:
        del_predict = np.delete(predict, training_idx)
        del_true_rss = np.delete(true_rss, training_idx)
        return mean_absolute_percentage_error(del_true_rss, del_predict)


def RMSE(true_rss, predict, training_idx):
    del_predict = np.delete(predict, training_idx)
    del_true_rss = np.delete(true_rss, training_idx)
    return mean_squared_error(del_true_rss, del_predict, squared=False)

def CRPS(true_rss, predict_mean, predict_var, training_idx):
    del_true_rss = np.delete(true_rss, training_idx)
    del_predict_mean = np.delete(predict_mean, training_idx)
    del_predict_var = np.delete(predict_var, training_idx)
    del_predict_sig = np.sqrt(del_predict_var)
    results = ps.crps_gaussian(del_true_rss, del_predict_mean, del_predict_sig)
    return results.mean()
    