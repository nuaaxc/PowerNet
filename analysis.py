from matplotlib import pyplot
import numpy as np
from utils import load_data_arima, load_energy_all
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from setting import APT_CSV


def autocorrelation_function_ACF():
    apt_name = 3
    train, _ = load_data_arima(APT_CSV % apt_name, '30T')
    plot_acf(train, lags=range(0, 100))
    pyplot.show()


def distribution_over_value():
    data = load_energy_all('data/Apt1_2016.csv')
    data = np.log(data)
    data = data.diff(24)
    data = data.dropna()
    data.hist()
    pyplot.show()


def decomposition():
    data = load_energy_all('data/Apt1_2016.csv')
    data = data['2016-11-01': '2016-11-30']
    result = seasonal_decompose(data, model='additive')
    result.plot()
    pyplot.show()


if __name__ == '__main__':
    autocorrelation_function_ACF()
    # distribution_over_value()
    # decomposition()

