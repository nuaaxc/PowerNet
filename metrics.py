
def mean_absolute_percentage_error(y_true, y_pred):
    import numpy as np
    mape = np.abs((y_true - y_pred) / y_true)
    mape = mape[~np.isinf(mape)]
    return np.mean(mape)
