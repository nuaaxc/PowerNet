import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 70)
# pd.set_option('display.max_rows', 200)

BASE_DIR = 'C:/Users/nuaax/Project/DeepPower/'
DATA_SET_DIR = BASE_DIR + 'data/2016/'
CACHE_DIR = BASE_DIR + 'data/'
MODEL_DIR = BASE_DIR + 'model/'
APT_CSV = DATA_SET_DIR + 'Apt%d_2016.csv'
GBT_RES_DIR = CACHE_DIR + '%s/res_gbt/'
SVM_RES_DIR = CACHE_DIR + '%s/res_svm/'
LSTM_RES_DIR = CACHE_DIR + '%s/res_lstm/'

FIG_DIR = CACHE_DIR + 'fig/'

TRAIN_START = "2016-03-01"
TRAIN_END = "2016-03-28"

TEST_START = "2016-03-29"
TEST_END = "2016-03-30"

# FREQ = 'H'
FREQ = '30T'


# SELECTED_FEAT = ['temperature', 'humidity', 'pressure', 'cloudCover', 'dewPoint', 'windSpeed', 'energy']
# SELECTED_FEAT = ['temperature', 'windSpeed', 'energy']
SELECTED_FEAT = ['energy', 'raw_energy', 'dewPoint']

ALL_FEAT = ['temperature',
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
            'precipProbability',
            'energy']


class PowerNetConfig:
    LSTM_DIM = 200
    DENSE_WEATHER_DIM = 256
    DENSE_MERGE_DIM = 256
    DROP_RATE = 0.1
    N_EPOCH = 500
    BATCH_SIZE = 128


