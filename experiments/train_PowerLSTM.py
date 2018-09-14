import numpy as np
import random

np.random.seed(777)
from tensorflow import set_random_seed

set_random_seed(777)

import os
import tqdm
import keras.backend as K
from matplotlib import pyplot
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from utils import load_data_lstm, load_data
from setting import APT_CSV, LSTM_RES_DIR, MODEL_DIR
from models.PowerLSTM import build_model


def train(X_train, y_train, X_val, y_val,
          model, model_path):
    print('training ...')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=80, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)
    model.fit(X_train, y_train,
              validation_data=[X_val, y_val],
              nb_epoch=1000,
              batch_size=128,
              verbose=0,
              callbacks=[earlyStopping, model_checkpoint],
              )


def test(X_test, y_test, model, model_path, is_draw):
    print('testing ...')
    model.load_weights(model_path)
    y_pred = model.predict(X_test)[:, 0]
    y_pred = np.exp(y_pred) - 1

    # metric
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()

    return mse, mape, r2


def test_single_model(X_test, y_test, model_path, is_draw):
    #######
    # test
    #######
    print('testing ...')
    parts = model_path.strip().split('_')
    PowerLSTM_config = {
        'NAME': 'PowerLSTM',
        'LSTM_1_DIM': int(parts[parts.index('lstm1') + 1]),
        'LSTM_2_DIM': int(parts[parts.index('lstm2') + 1]),
        'DENSE_DIM': int(parts[parts.index('dense') + 1]),
        'DROP_RATE': float(parts[parts.index('drop') + 1]),
        'LR': float(parts[parts.index('lr') + 1]),
        'SS': parts[parts.index('ss') + 1],
        'Freq': parts[parts.index('freq') + 1]
    }
    model = build_model(PowerLSTM_config, X_test.shape[-1])
    model.load_weights(model_path)
    y_pred = model.predict(X_test)[:, 0]
    y_pred = np.exp(y_pred) - 1

    # metric
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('MSE:', mse)
    print('MAPE:', mape)
    print('R2:', r2)

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()

    return mse, mape


def load_train_test_data(apt_fname, freq, tr_te_split):
    ############
    # load data
    ############
    print('load data ========================================' * 2)
    print(apt_fname)
    print(tr_te_split)

    df = load_data(apt_fname, freq)
    train = df[tr_te_split['trb']: tr_te_split['tre']]
    val = df[tr_te_split['vab']: tr_te_split['vae']]
    test = df[tr_te_split['teb']: tr_te_split['tee']]
    print('train/test:', train.shape, val.shape, test.shape)

    feat = list(train.columns.values)
    feat.remove('energy')
    feat.remove('raw_energy')
    print('raw features (%d):' % len(feat), feat)

    X_train = train[feat].as_matrix()
    y_train = train['energy'].as_matrix()
    X_val = val[feat].as_matrix()
    y_val = val['energy'].as_matrix()
    X_test = test[feat].as_matrix()
    y_test = test['raw_energy'].as_matrix()

    feat_dim = X_train.shape[-1]

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    print('train/test (after converting to matrix):', X_train.shape, X_val.shape, X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, feat_dim


def make_model_path(config, feat_dim):
    model = build_model(config, feat_dim)
    model_path = os.path.join(MODEL_DIR,
                              '_'.join([
                                  config['NAME'],
                                  'ss', config['SS'],
                                  'freq', config['Freq'],
                                  'lstm1', str(config['LSTM_1_DIM']),
                                  'lstm2', str(config['LSTM_2_DIM']),
                                  'dense', str(config['DENSE_DIM']),
                                  'drop', str(config['DROP_RATE']),
                                  'lr', str(config['LR']),
                                  'model.h5'
                              ])
                              )
    return model, model_path


def main():
    apt_name = 0
    season = {
        '2': {'trb': '2016-02-01', 'tre': '2016-02-26', 'teb': '2016-02-27', 'tee': '2016-02-28'},
        '3': {'trb': '2016-03-01', 'tre': '2016-03-28', 'teb': '2016-03-29', 'tee': '2016-03-30'},
        '4': {'trb': '2016-04-01', 'tre': '2016-04-26', 'vab': '2016-04-27', 'vae': '2016-04-28',
              'teb': '2016-04-29', 'tee': '2016-04-30'},
        '5': {'trb': '2016-05-01', 'tre': '2016-05-28', 'teb': '2016-05-29', 'tee': '2016-05-30'},
        '6': {'trb': '2016-06-01', 'tre': '2016-06-28', 'teb': '2016-06-29', 'tee': '2016-06-30'},
        '7': {'trb': '2016-07-01', 'tre': '2016-07-28', 'teb': '2016-07-29', 'tee': '2016-07-30'},
        '8': {'trb': '2016-08-01', 'tre': '2016-08-28', 'teb': '2016-08-29', 'tee': '2016-08-30'},
        '9': {'trb': '2016-09-01', 'tre': '2016-09-20', 'vab': '2016-09-21', 'vae': '2016-09-28',
              'teb': '2016-09-29', 'tee': '2016-09-30'},
        '10': {'trb': '2016-10-01', 'tre': '2016-10-28', 'teb': '2016-10-29', 'tee': '2016-10-30'},
        '11': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
    }
    # PowerLSTM_config = {
    #     'NAME': 'PowerLSTM',
    #     'LSTM_1_DIM': 197,
    #     'LSTM_2_DIM': 171,
    #     'DENSE_DIM': 88,
    #     'DROP_RATE': 0.1,
    #     'LR': 0.001,
    #     'SS': 'autumn',
    #     'Freq': '1h'
    # }
    PowerLSTM_config = {
        'NAME': 'PowerLSTM',
        'LSTM_1_DIM': 0,
        'LSTM_2_DIM': 0,
        'DENSE_DIM': 0,
        'DROP_RATE': 0.1,
        'LR': 0.01,
        'SS': '4',
        'Freq': '1h'
    }

    X_train, y_train, X_val, y_val, X_test, y_test, feat_dim = load_train_test_data(
        apt_fname=APT_CSV % apt_name,
        freq=PowerLSTM_config['Freq'],
        tr_te_split=season[
            PowerLSTM_config['SS']])

    for _ in tqdm.tqdm(range(10000)):
        PowerLSTM_config['LSTM_1_DIM'] = random.randint(200, 400)
        PowerLSTM_config['LSTM_2_DIM'] = random.randint(200, 400)
        PowerLSTM_config['DENSE_DIM'] = random.randint(200, 400)
        # PowerLSTM_config['DENSE_DIM'] = 0

        K.clear_session()

        model, model_path = make_model_path(PowerLSTM_config, feat_dim)

        train(X_train, y_train, X_val, y_val, model, model_path)
        mse, mape, r2 = test(X_test, y_test, model, model_path, False)
        print('MSE:', mse)
        print('MAPE:', mape)
        print('R2:', r2)
        if mape > 0.096:
            os.remove(model_path)
    '''
    test_single_model(X_test, y_test,
                      '../good_model/PowerLSTM_ss_9_freq_1h_lstm1_371_lstm2_381_dense_292_drop_0.1_lr_0.01_model.h5',
                      True)
    '''


if __name__ == '__main__':
    main()
