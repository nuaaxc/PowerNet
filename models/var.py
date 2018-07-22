import numpy as np
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
from utils import load_data_var
from metrics import mean_absolute_percentage_error


def train_predict():
    train, test = load_data_var()
    print len(train), len(test)
    train = train.as_matrix()
    test = test.as_matrix()
    predictions = list()
    for t in range(len(test)):
        model = VAR(train)
        model_fit = model.fit(3)
        lag_order = model_fit.k_ar
        output = model_fit.forecast(train[-lag_order:], lag_order)
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t, np.newaxis]
        train = np.concatenate((train, obs), axis=0)
        print('predicted=%s, expected=%s' % (yhat, obs))
    predictions = np.array(predictions)

    e_pred = predictions[:, -1]
    e_test = test[:, -1]
    error = mean_squared_error(e_test, e_pred)
    print('Test MSE: %.3f' % error)
    print 'MAPE:', mean_absolute_percentage_error(e_test, e_pred)
    # plot
    pyplot.plot(e_test)
    pyplot.plot(e_pred, color='red')
    pyplot.show()


if __name__ == '__main__':
    train_predict()
