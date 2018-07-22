import os
import numpy as np
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils import load_matrix
from setting import DATA_SET_DIR, MODEL_DIR
from models.PowerNet import build_model as powernet
from models.RNN import build_model as rnn

from sklearn.metrics import mean_squared_error, mean_absolute_error
from metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

season = {
    'autumn': {'trb': '2016-02-01', 'tre': '2016-02-25',
               'vab': '2016-02-26', 'vae': '2016-02-27',
               'teb': '2016-02-28', 'tee': '2016-02-29'},
    'summer': {'trb': '2016-06-01', 'tre': '2016-08-17',
               'vab': '2016-08-18', 'vae': '2016-08-24',
               'teb': '2016-08-25', 'tee': '2016-08-31'},
    'spring': {'trb': '2016-04-01', 'tre': '2016-04-26',
               'vab': '2016-04-27', 'vae': '2016-04-28',
               'teb': '2016-04-29', 'tee': '2016-04-30'},
    'winter': {'trb': '2016-11-01', 'tre': '2016-11-26',
               'vab': '2016-11-27', 'vae': '2016-07-28',
               'teb': '2016-11-29', 'tee': '2016-11-30'},
}


def make_model_path(config):
    return os.path.join(MODEL_DIR,
                        '_'.join([
                            config['NAME'],
                            'lstm', str(config['LSTM_DIM']),
                            'denseW', str(config['DENSE_WEATHER_DIM']),
                            'denseM', str(config['DENSE_PREDICTION_DIM']),
                            'drop', str(config['DROP_RATE']),
                            'model.h5'
                        ])
                        )


def training(X_train_energy, X_train_weather, y_train,
             X_valid_energy, X_valid_weather, y_valid,
             model, model_path, training_config):
    # train
    print('Training ...')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True)

    model.fit(x=[X_train_energy, X_train_weather],
              y=y_train,
              validation_data=([X_valid_energy, X_valid_weather], y_valid),
              epochs=training_config['N_EPOCH'],
              batch_size=training_config['BATCH_SIZE'],
              verbose=1,
              shuffle=True,
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


def prepare(freq, ss, model_config, build_model):
    # load data
    (X_train_weather, X_train_energy, y_train,
     X_valid_weather, X_valid_energy, y_valid,
     X_test_weather, X_test_energy, y_test) = load_matrix(DATA_SET_DIR + 'SUM_%d_%s_2016.pkl' % (114, freq),
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

    # config model
    model = build_model(model_config, X_train_energy.shape[1], X_train_weather.shape[1])
    model_path = make_model_path(model_config)

    return (X_train_energy, X_train_weather, y_train,
            X_valid_energy, X_valid_weather, y_valid,
            X_test_energy, X_test_weather, y_test,
            model, model_path)


if __name__ == '__main__':
    PowerNet_config = {
        'NAME': 'PowerNet',
        'LSTM_DIM': 512,
        'DENSE_WEATHER_DIM': 256,
        'DENSE_ATTENTION_DIM': 256,
        'DENSE_PREDICTION_DIM': 256,
        'DROP_RATE': 0.1,
        'LR': 0.0001,
    }
    training_config = {
        'N_EPOCH': 500,
        'BATCH_SIZE': 256,
    }
    (X_train_energy, X_train_weather, y_train,
     X_valid_energy, X_valid_weather, y_valid,
     X_test_energy, X_test_weather, y_test,
     model, model_path) = prepare('1h', 'summer', PowerNet_config, powernet)

    training(X_train_energy, X_train_weather, y_train,
             X_valid_energy, X_valid_weather, y_valid,
             model, model_path, training_config)

    test(X_test_energy, X_test_weather, y_test,
         model, model_path, is_draw=True)
