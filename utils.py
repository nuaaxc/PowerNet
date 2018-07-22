from pandas import read_csv
from matplotlib import pyplot
from os.path import basename
import numpy as np
import pandas as pd
from pandas import datetime
from datetime import timedelta
from setting import DATA_SET_DIR, TRAIN_START, TRAIN_END, TEST_START, TEST_END, SELECTED_FEAT, FREQ, ALL_FEAT
from sklearn.preprocessing import scale


def load_matrix(apt_name, freq, data_split):
    df = load_data(apt_name, freq)
    train = df[data_split['trb']: data_split['tre']]
    valid = df[data_split['vab']: data_split['vae']]
    test = df[data_split['teb']: data_split['tee']]
    print('train/valid/test:', train.shape, valid.shape, test.shape)

    feat = list(train.columns.values)
    feat.remove('energy')
    feat.remove('raw_energy')
    print('raw features (%d):' % len(feat), feat)

    feat_weather = ['temperature', 'icon', 'humidity', 'visibility',
                    'summary', 'apparentTemperature', 'pressure',
                    'windSpeed', 'cloudCover', 'windBearing',
                    'precipIntensity', 'dewPoint', 'precipProbability',
                    'week_month', 'day_of_week', 'hour', 'part_of_day', 'is_weekend']
    feat_energy = feat[len(feat_weather):]

    X_train_weather = train[feat_weather].as_matrix()
    X_train_energy = train[feat_energy].as_matrix()
    X_train_energy = np.expand_dims(X_train_energy, axis=-1)
    y_train = train['energy'].as_matrix()

    X_valid_weather = valid[feat_weather].as_matrix()
    X_valid_energy = valid[feat_energy].as_matrix()
    X_valid_energy = np.expand_dims(X_valid_energy, axis=-1)
    y_valid = valid['energy'].as_matrix()

    X_test_weather = test[feat_weather].as_matrix()
    X_test_energy = test[feat_energy].as_matrix()
    X_test_energy = np.expand_dims(X_test_energy, axis=-1)
    y_test = test['raw_energy'].as_matrix()

    return (X_train_weather, X_train_energy, y_train,
            X_valid_weather, X_valid_energy, y_valid,
            X_test_weather, X_test_energy, y_test)


def preprocessing(df):
    # add temporal-based features
    dt = df.index.tolist()
    df['week_month'] = [d.days_in_month for d in dt]
    df['day_of_week'] = [d.dayofweek for d in dt]
    df['hour'] = [d.hour / 3 for d in dt]
    df['part_of_day'] = list(map(lambda x: 0 if x.hour in range(6, 19) else 1, dt))
    df['is_weekend'] = [int(d.dayofweek in set([5, 6])) for d in dt]
    # for i in range(1, 4) + range(22, 28) + range(34, 37) + range(46, 50):
    for i in range(1, 50):
        df['energy-%s' % i] = df.shift(i)['energy']

    # categorical to numeric
    df['icon'] = df['icon'].astype('category')
    df['summary'] = df['summary'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    # feature standardization
    feat_standard = ['temperature',
                     'humidity',
                     'visibility',
                     'apparentTemperature',
                     'pressure',
                     'windSpeed',
                     'cloudCover',
                     'windBearing',
                     'precipIntensity',
                     'dewPoint',
                     'precipProbability']
    df[feat_standard] = scale(df[feat_standard])
    return df


def load_data(apt_fname, freq):

    energy = None
    # 1) read data
    if basename(apt_fname).startswith('Apt'):
        energy = read_csv(apt_fname,
                          header=None,
                          names=['energy'],
                          index_col=0,
                          squeeze=True,
                          date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        energy = energy.resample(freq).mean()
    elif basename(apt_fname).startswith('Mean_seed') \
            or basename(apt_fname).startswith('SUM_'):
        energy = pd.read_pickle(apt_fname)
        energy = energy.rename('energy')

    log_energy = np.log(energy+1)
    # s_series = s_series.diff()
    df_energy = pd.DataFrame({'raw_energy': energy, 'energy': log_energy})

    df_weather = read_csv(DATA_SET_DIR + 'apartment2016.csv',
                          usecols=['temperature',
                                   'icon',
                                   'humidity',
                                   'visibility',
                                   'summary',
                                   'apparentTemperature',
                                   'pressure',
                                   'windSpeed',
                                   'cloudCover',
                                   'time',
                                   'windBearing',
                                   'precipIntensity',
                                   'dewPoint',
                                   'precipProbability'],
                          index_col=9,
                          squeeze=True,
                          date_parser=lambda x: datetime.fromtimestamp(int(x)) - timedelta(hours=5),
                          parse_dates=['time'])
    df_weather = df_weather.dropna()
    df = df_energy.join(df_weather, how='inner').bfill()
    df = df.dropna()

    # 2) pre-processing
    df = preprocessing(df)

    return df


def generate_dt_feature(df):
    """
    week of month
    day of week
    is_weekend
    day_part: morning, afternoon, evening, night
    """

    def part_of_day(x):
        """
        0: morning [8,9,10,11,12]
        1: afternoon [13, 14, 15, 16, 17]
        2: evening [18, 19, 20, 21, 22]
        3: night [23, 0, ..., 7]
        """
        x = x.hour
        if x in range(8, 13):
            return 0
        elif x in range(13, 18):
            return 1
        elif x in range(18, 23):
            return 2
        else:
            return 3

    dt = df.index.tolist()
    # df['week_month'] = [d.days_in_month for d in dt]
    # df['day_of_week'] = [d.dayofweek for d in dt]
    # df['hour'] = [d.hour/3 for d in dt]
    # df['part_of_day'] = map(part_of_day, dt)
    # df['is_weekend'] = [d.dayofweek in set([5, 6]) for d in dt]

    df['energy-%s' % 1] = df.shift(1)['energy']
    df['energy-%s' % 2] = df.shift(2)['energy']
    df['energy-%s' % 3] = df.shift(3)['energy']
    df['energy-%s' % 24] = df.shift(24)['energy']
    df['energy-%s' % 48] = df.shift(48)['energy']

    # for i in range(1, 4) + range(21, 25) + range(46, 50):
    #     df['energy-%s' % i] = df.shift(i)['energy']
    return df


def load_energy(apt):
    from setting import APT_CSV

    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    series = read_csv(APT_CSV % apt,
                      header=None,
                      names=['%d' % apt],
                      index_col=0,
                      squeeze=True,
                      date_parser=parser)
    series = series['2016-01-01': '2016-12-01']
    return series


def load_weather():
    def parser(x):
        return datetime.fromtimestamp(int(x)) - timedelta(hours=4)

    df = read_csv('/Users/kevin/PycharmProjects/TimeSeries/data/apartment2016.csv',
                  usecols=['temperature', 'humidity', 'pressure', 'windSpeed',
                           'cloudCover', 'time', 'precipIntensity', 'dewPoint'],
                  index_col=5, squeeze=True, date_parser=parser, parse_dates=['time'])
    return df


def load_data_var():
    energy = load_energy()
    weather = load_weather()

    data = weather.join(DataFrame(energy), how='inner')

    data = data[SELECTED_FEAT]

    # data = data.diff()
    data = data.dropna()

    train = data[TRAIN_START: TRAIN_END]
    test = data[TEST_START: TEST_END]

    return train, test


def load_data_feature_selection():
    energy = load_energy()
    del energy['raw_energy']

    weather = read_csv('/Users/kevin/PycharmProjects/TimeSeries/data/apartment2016.csv',
                       usecols=['temperature',
                                'icon',
                                'humidity',
                                'visibility',
                                'summary',
                                'apparentTemperature',
                                'pressure',
                                'windSpeed',
                                'cloudCover',
                                'time',
                                'windBearing',
                                'precipIntensity',
                                'dewPoint',
                                'precipProbability'],
                       index_col=9,
                       squeeze=True,
                       date_parser=lambda x: datetime.fromtimestamp(int(x)) - timedelta(hours=4),
                       parse_dates=['time'])

    data = weather.join(pd.DataFrame(energy), how='outer').bfill()
    data = data.dropna()

    data = generate_dt_feature(data)

    data['icon'] = data['icon'].astype('category')
    data['summary'] = data['summary'].astype('category')
    data['is_weekend'] = data['is_weekend'].astype('category')
    cat_columns = data.select_dtypes(['category']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: x.cat.codes)

    train = data[TRAIN_START: TRAIN_END]

    return train


def load_data_svm():
    energy = load_energy()
    weather = load_weather()

    data = weather.join(energy, how='outer').bfill()

    data = data[SELECTED_FEAT]

    data = data.dropna()

    data = generate_dt_feature(data)

    train = data[TRAIN_START: TRAIN_END]
    test = data[TEST_START: TEST_END]

    return train, test


def load_energy_all(filename):
    data = read_csv(filename, header=None, names=['energy'],
                    index_col=0, squeeze=True, date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data = data.resample(FREQ).mean()
    data = data.dropna()
    return data


def load_data_arima(filename, freq):
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    data = read_csv(filename, header=None, names=['energy'], index_col=0, squeeze=True, date_parser=parser)
    data = data.resample(freq).mean()
    # data = np.log(data)
    data = data.dropna()

    train = data['2016-07-01': '2016-07-30']
    test = data['2016-10-01': '2016-10-01']

    return train, test


def load_data_lstm(filename):
    def parser(x):
        return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    data = read_csv(filename, header=None, names=['energy'], index_col=0, squeeze=True, date_parser=parser)
    data = data.resample(FREQ).mean()
    raw_data = data
    data = np.log(data)
    data = data.dropna()

    data = pd.DataFrame({'x': data.values}, index=data.index)
    data['raw_energy'] = raw_data
    data['x-1'] = data.shift(1)['x']
    data['x-2'] = data.shift(2)['x']
    data['x-3'] = data.shift(3)['x']
    data['x-24'] = data.shift(24)['x']
    data['x-48'] = data.shift(48)['x']

    data = data[['raw_energy', 'x-48', 'x-24', 'x-3', 'x-2', 'x-1', 'x']]

    train = data[TRAIN_START: TRAIN_END]
    test = data[TEST_START: TEST_END]

    return train, test


if __name__ == '__main__':
    energy2matrix()


