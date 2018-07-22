import dismalpy as dp
import numpy as np
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from utils import load_data_var
import statsmodels.api as sm
import pandas as pd

lag_order = 1


def train_predict():
    """
    (3,3): 0.154
    (3,2): 0.152
    (3,1): 0.145
    """
    train, test = load_data_var()
    print len(train), len(test)
    train = train.as_matrix()
    test = test.as_matrix()
    predictions = list()
    for t in range(len(test)):

        model = sm.tsa.VARMAX(train, order=(3, 1))
        model_fit = model.fit(maxiter=10)
        output = model_fit.forecast()
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
    # plot
    pyplot.plot(e_test)
    pyplot.plot(e_pred, color='red')
    pyplot.show()


def test():
    import statsmodels.api as sm
    import pandas as pd
    dta = pd.read_stata('/Users/kevin/PycharmProjects/TimeSeries/data/lutkepohl2.dta')
    dta.index = dta.qtr
    endog = dta.ix['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]

    print endog
    mod = sm.tsa.VARMAX(endog[['dln_inv', 'dln_inc']], order=(1, 1))
    res = mod.fit(maxiter=1000)
    print(res.summary())


if __name__ == '__main__':
    train_predict()  #[MSE 0.145]
    # test()




