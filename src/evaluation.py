import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

def mae(a,b):
    return mean_absolute_error(a,b)

def mape(a,b):
    return np.mean(np.abs((a-b)/a))*100
