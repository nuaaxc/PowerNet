import os
import tqdm
import keras.backend as K
import random
import numpy as np
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint, EarlyStopping

from utils import load_matrix
from setting import MODEL_DIR, APT_CSV
from models.PowerNet import build_model as powernet

from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

season = {
    '2': {'trb': '2016-02-01', 'tre': '2016-02-26', 'teb': '2016-02-27', 'tee': '2016-02-28'},
    '3': {'trb': '2016-03-01', 'tre': '2016-03-28', 'teb': '2016-03-29', 'tee': '2016-03-30'},
    '4': {'trb': '2016-04-01', 'tre': '2016-04-26', 'vab': '2016-04-27', 'vae': '2016-04-28',
          'teb': '2016-04-29', 'tee': '2016-04-30'},
    '5': {'trb': '2016-05-01', 'tre': '2016-05-28', 'teb': '2016-05-29', 'tee': '2016-05-30'},
    '6': {'trb': '2016-06-01', 'tre': '2016-06-20', 'vab': '2016-06-21', 'vae': '2016-06-28',
          'teb': '2016-06-29', 'tee': '2016-06-30'},
    '7': {'trb': '2016-07-01', 'tre': '2016-07-28', 'teb': '2016-07-29', 'tee': '2016-07-30'},
    '8': {'trb': '2016-08-01', 'tre': '2016-08-28', 'teb': '2016-08-29', 'tee': '2016-08-30'},
    '9': {'trb': '2016-09-01', 'tre': '2016-09-20', 'vab': '2016-09-21', 'vae': '2016-09-28',
          'teb': '2016-09-29', 'tee': '2016-09-30'},
    '10': {'trb': '2016-10-01', 'tre': '2016-10-28', 'teb': '2016-10-29', 'tee': '2016-10-30'},
    '11': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
}


def make_model_path(config, energy_dim, weather_dim):
    model = powernet(config, energy_dim, weather_dim)
    model_path = os.path.join(MODEL_DIR,
                              '_'.join([
                                  config['NAME'],
                                  'ss', config['SS'],
                                  'freq', config['Freq'],
                                  'lstm', str(config['LSTM_DIM']),
                                  'denseW', str(config['DENSE_WEATHER_DIM']),
                                  'denseM', str(config['DENSE_PREDICTION_DIM']),
                                  'drop', str(config['DROP_RATE']),
                                  'lr', str(config['LR']),
                                  'model.h5'
                              ])
                              )
    return model, model_path


def training(X_train_energy, X_train_weather, y_train,
             X_valid_energy, X_valid_weather, y_valid,
             model, model_path):
    # train
    print('Training ...')

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)

    model.fit(x=[X_train_energy, X_train_weather],
              y=y_train,
              validation_data=([X_valid_energy, X_valid_weather], y_valid),
              epochs=1000,
              batch_size=128,
              verbose=0,
              callbacks=[early_stopping, model_checkpoint]
              )


def test(X_test_energy, X_test_weather, y_test,
         model, model_path, is_draw):
    model.load_weights(model_path)

    print('Testing ...')
    y_pred = teaching(model, X_test_energy, X_test_weather)
    # y_pred = teaching_no(model, X_test_energy, X_test_weather)

    print('Evaluating ...')
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('MSE:', mse)
    print('MAPE:', mape)
    print('R2:', r2)

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='green')
        pyplot.show()

    return mse, mape, r2


def teaching(model, X_enery, X_weather):
    y_pred = model.predict([X_enery, X_weather],
                           verbose=1)[:, 0]
    y_pred = np.exp(y_pred) - 1
    return y_pred


def teaching_no(model, X_enery, X_weather):
    y_pred = []
    n_samples = X_enery.shape[0]
    n_time_steps = X_enery.shape[1]
    n_weather_features = X_weather.shape[1]
    x_energy = (np.copy(X_enery[0])).reshape(1, n_time_steps, 1)  # (batch_size, seq_length, scalar)
    for i in range(n_samples):
        predict_result = model.predict([x_energy, X_weather[i].reshape(1, n_weather_features)],  # (batch_size, scalar)
                                       verbose=1)[:, 0]
        predict_result = np.exp(predict_result) - 1
        predict_value = predict_result[0]

        # save result and update input
        y_pred.append(predict_value)
        for j in range(len(x_energy[0]) - 1):
            x_energy[0][j] = x_energy[0][j + 1]
        x_energy[0][-1] = predict_value
    y_pred = np.asarray(y_pred)
    return y_pred


def prepare(freq, ss):
    # load data
    (X_train_weather, X_train_energy, y_train,
     X_valid_weather, X_valid_energy, y_valid,
     X_test_weather, X_test_energy, y_test) = load_matrix(APT_CSV % apt_name,
                                                          freq,
                                                          season[ss])

    print('Training data:')
    print('\tX_weather:', X_train_weather.shape)
    print('\tX_energy:', X_train_energy.shape)
    print('\ty:', y_train.shape)

    print('Validation data:')
    print('\tX_weather:', X_valid_weather.shape)
    print('\tX_energy:', X_valid_energy.shape)
    print('\ty:', y_valid.shape)

    energy_dim, weather_dim = X_train_energy.shape[1], X_train_weather.shape[1]

    return (X_train_energy, X_train_weather, y_train,
            X_valid_energy, X_valid_weather, y_valid,
            X_test_energy, X_test_weather, y_test,
            energy_dim, weather_dim)


if __name__ == '__main__':
    apt_name = 0
    PowerNet_config = {
        'NAME': 'PowerNet',
        'LSTM_DIM': 0,
        'DENSE_WEATHER_DIM': 0,
        'DENSE_ATTENTION_DIM': 0,
        'DENSE_PREDICTION_DIM': 0,
        'DROP_RATE': 0.2,
        'LR': 0.0001,
        'SS': '6',
        'Freq': '1h'
    }

    (X_train_energy, X_train_weather, y_train,
     X_valid_energy, X_valid_weather, y_valid,
     X_test_energy, X_test_weather, y_test,
     energy_dim, weather_dim) = prepare(PowerNet_config['Freq'], PowerNet_config['SS'])

    for _ in tqdm.tqdm(range(10000)):

        PowerNet_config['LSTM_DIM'] = random.randint(100, 200)
        PowerNet_config['DENSE_WEATHER_DIM'] = random.randint(100, 200)
        PowerNet_config['DENSE_ATTENTION_DIM'] = random.randint(100, 200)
        PowerNet_config['DENSE_PREDICTION_DIM'] = random.randint(100, 200)

        K.clear_session()

        model, model_path = make_model_path(PowerNet_config, energy_dim, weather_dim)

        training(X_train_energy, X_train_weather, y_train,
                 X_valid_energy, X_valid_weather, y_valid,
                 model, model_path)

        mse, mape, r2 = test(X_test_energy, X_test_weather, y_test,
                             model, model_path, is_draw=False)

        if mape > 0.12:
            os.remove(model_path)
