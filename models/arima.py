from matplotlib import pyplot
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from utils import load_data_arima
from metrics import mean_absolute_percentage_error


def train_predict():
    train, test = load_data_arima('/Users/kevin/PycharmProjects/TimeSeries/data/Apt1_2016.csv')
    print len(train), len(test)
    model = ARIMA(train, order=(5, 1, 2))
    model_fit = model.fit(disp=0)
    predictions = model_fit.forecast(steps=len(test))[0]
    # predictions = np.roll(predictions, -1)
    test = test.values
    error = mean_squared_error(test, predictions)
    print 'MSE:', error
    print 'MAPE:', mean_absolute_percentage_error(test, predictions)

    # plot
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()


if __name__ == '__main__':
    # analysis()
    train_predict()



