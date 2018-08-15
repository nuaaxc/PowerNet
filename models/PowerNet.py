# LSTM for international airline passengers problem with regression framing
import os
import sys
import time
import numpy as np
import keras
from matplotlib import pyplot
from pandas import read_csv
import pickle
from keras.models import Model
from keras.layers import Input, Dense, LSTM, BatchNormalization, concatenate
import keras.backend as K
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from utils import load_data
from setting import APT_CSV, LSTM_RES_DIR, PowerNetConfig
import h5py
from keras.models import load_model
from feature_selection import feature_selection

np.random.seed(42)
'''np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)'''



def build_model(model_config, weather_dim, energy_dim, learning_rate=0.0001):
    lstm_dim = model_config.LSTM_DIM
    dense_weather_dim = model_config.DENSE_WEATHER_DIM
    dense_merge_dim = model_config.DENSE_MERGE_DIM
    drop_rate = model_config.DROP_RATE

    # Energy part
    e = Input(shape=(energy_dim, 1,), dtype='float32', name='energy_input')
    # lstm = LSTM(lstm_dim, return_sequences=True, dropout=drop_rate)(e)
    lstm = LSTM(lstm_dim, dropout=drop_rate, return_sequences=True)(e)
    lstm = LSTM(lstm_dim, dropout=drop_rate, return_sequences=True)(lstm)
    lstm = LSTM(lstm_dim, dropout=drop_rate)(lstm)
    lstm = BatchNormalization()(lstm)

    # Weather part
    w = Input(shape=(weather_dim,), dtype='float32', name='weather_input')
    dense_weather = Dense(dense_weather_dim, activation='relu')(w)
    dense_weather = Dense(dense_weather_dim, activation='relu')(dense_weather)

    # merge layer
    m = concatenate([lstm, dense_weather])
    dense_merge = Dense(dense_merge_dim, activation='relu')(m)

    # prediction layer
    prediction = Dense(1)(dense_merge)

    model = Model(inputs=[e, w], outputs=prediction)

    optimizer = keras.optimizers.Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.0)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)

    model.summary()

    return model


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


def run_model_based_prediction(model_config, X_train_weather, X_train_energy, y_train, X_valid_weather,
                               X_valid_energy, y_valid, X_test_weather, X_test_energy, y_test):
    # X_train_weather (550, 18), X_train_energy (550, 49, 1), y_train (550,)
    # X_valid_weather (44, 18),  X_valid_energy (44, 49, 1),  y_valid (44,)
    # X_test_weather  (48, 18),  X_test_energy  (48, 49, 1),  y_test  (48,)
    final_pred = np.array([], dtype=float)
    original_y_test = np.copy(y_test)
    mses = []
    mapes = []
    for _ in range(len(X_test_weather)):
        ########
        # train
        ########
        print('--- training ...')
        model = build_model(model_config, X_train_weather.shape[1], X_train_energy.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model.fit(x=[X_train_energy, X_train_weather],
                  y=y_train,
                  validation_data=([X_valid_energy, X_valid_weather], y_valid),
                  epochs=model_config.N_EPOCH,
                  batch_size=model_config.BATCH_SIZE,
                  verbose=1,
                  shuffle=False,
                  callbacks=[early_stopping]
                  )

        #######
        # test
        #######
        print('--- testing ...')
        y_pred = model.predict([X_test_energy[0].reshape(1, 49, 1), X_test_weather[0].reshape(1, 18)],
                               verbose=1)[:, 0]
        y_pred = np.exp(y_pred) - 1
        final_pred = np.append(final_pred, y_pred)
        if len(final_pred) == len(original_y_test):
            break

        mse = mean_squared_error(np.asarray([y_test[0]]), np.asarray([y_pred[0]]))
        mses.append(mse)
        mape = mean_absolute_percentage_error(np.asarray([y_test[0]]), np.asarray([y_pred[0]]))
        mapes.append(mape)

        # Update
        # train
        X_train_weather = np.concatenate((X_train_weather[1:], X_valid_weather[0].reshape(1, 18)))
        X_train_energy = np.concatenate((X_train_energy[1:], X_valid_energy[0].reshape(1, 49, 1)))
        y_train = np.concatenate((y_train[1:], y_valid[0].reshape(1, )))
        # valid
        X_valid_weather = np.concatenate((X_valid_weather[1:], X_test_weather[0].reshape(1, 18)))
        X_valid_energy = np.concatenate((X_valid_energy[1:], X_test_energy[0].reshape(1, 49, 1)))
        y_valid = np.concatenate((y_valid[1:], y_test[0].reshape(1, )))
        # test
        X_test_weather = X_test_weather[1:]
        X_test_energy = X_test_energy[1:]
        y_test = y_test[1:]

    print("mse")
    print(mses)
    print(sum(mses) / len(mses))
    print("mape")
    print(mapes)
    print(sum(mapes) / len(mapes))

    return original_y_test, final_pred


def load_apt_data(apt_name, data_split, freq='1h'):
    ############
    # load data
    ############
    print('--- loading data from %s ...' % apt_name)
    print('--- train/test split:', data_split)

    (X_train_weather, X_train_energy, y_train,
     X_valid_weather, X_valid_energy, y_valid,
     X_test_weather, X_test_energy, y_test) = load_matrix(apt_name, freq, data_split)

    print('--- training data shape:')
    print('------ X_weather:', X_train_weather.shape)
    print('------ X_energy:', X_train_energy.shape)
    print('------ y:', y_train.shape)

    print('--- validation data shape:')
    print('------ X_weather:', X_valid_weather.shape)
    print('------ X_energy:', X_valid_energy.shape)
    print('------ y:', y_valid.shape)

    print('--- test data shape:')
    print('------ X_weather:', X_test_weather.shape)
    print('------ X_energy:', X_test_energy.shape)
    print('------ y:', y_test.shape)

    return X_train_weather, X_train_energy, y_train, \
           X_valid_weather, X_valid_energy, y_valid, \
           X_test_weather, X_test_energy, y_test


def run_model(data,
              res_file_pref, model_config, set_index,
              is_draw=False, is_save=False, learning_rate=0.0001):

    X_train_weather, X_train_energy, y_train, \
    X_valid_weather, X_valid_energy, y_valid, \
    X_test_weather, X_test_energy, y_test = data

    ########
    # train
    ########

    K.clear_session()

    print('--- training ...')

    model = build_model(model_config, X_train_weather.shape[1], X_train_energy.shape[1], learning_rate)
    #model = load_model_from_file("mse", res_file_pref)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=[X_train_energy, X_train_weather],
              y=y_train,
              validation_data=([X_valid_energy, X_valid_weather], y_valid),
              epochs=model_config.N_EPOCH,
              batch_size=model_config.BATCH_SIZE,
              verbose=1,
              shuffle=False,
              callbacks=[early_stopping]
              )

    
    #Predict using reported values
    #######
    # test
    #######
    print('--- testing ...')
    y_pred = model.predict([X_test_energy, X_test_weather],
                           verbose=1)[:, 0]
    y_pred = np.exp(y_pred)-1

    #############
    # evaluation
    #############
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('MSE:', mse)
    print('MAPE:', mape)
    print('R2:', r2)
    if is_save:
        #print('save result to file ...')
        #pickle.dump(
        #    {'y_test': y_test, 'y_pred': y_pred},
        #    open(res_file_pref + '_mse%.8f_mape%.4f_r%.4f_ly1_%d_ly2_%d.pkl' %
        #         (mse, mape, r2, model_config.LSTM_DIM, model_config.LSTM_DIM), 'wb'))
        #print('saved at '+ res_file_pref + '_mse%.8f_mape%.4f_r%.4f_ly1_%d_ly2_%d.pkl' %
        #         (mse, mape, r2, model_config.LSTM_DIM, model_config.LSTM_DIM))
        
        print('save result to file ...')
        with open(res_file_pref, 'a+') as f:
            f.write("set_index:{}\n".format(set_index))
            f.write("lstm_dim:{}\ndense_weather_dim:{}\ndense_merge_dim:{}\ndropout_rate:{}\nlearning_rate:{}\n"
                    .format(model_config.LSTM_DIM, model_config.DENSE_WEATHER_DIM, model_config.DENSE_MERGE_DIM,
                            model_config.DROP_RATE, learning_rate))
            f.write("y_test:{}\n".format(','.join([str(v) for v in y_test])))
            f.write("y_predict:{}\n".format(','.join([str(v) for v in y_pred])))
            f.write("Deviation:{}\n".format(
                ','.join([str(abs(test - pred) / test) for test, pred in zip(y_test, y_pred)])))
            f.write("MSE:{}\nMAPE:{}\nR2:{}\n\n".format(mse, mape, r2))
    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()

    # save models for each parameter set
    save_model_to_file(model, str(set_index), res_file_pref)

    return mse, mape


    '''
    print("\nHourly Prediction")
    y_predict_hourly = []
    x_test_energy_hourly = (np.copy(X_test_energy[0])).reshape(1, 49, 1)
    for i in range(len(y_test)):
        # predict next day (predict_result is a list with one value in this case)
        predict_result = model.predict([x_test_energy_hourly, X_test_weather[i].reshape(1, 18)],
                                       verbose=1)[:, 0]
        predict_result = np.exp(predict_result) - 1
        predict_value = predict_result[0]

        # save result and update input
        y_predict_hourly.append(predict_value)
        for j in range(len(x_test_energy_hourly[0]) - 1):
            x_test_energy_hourly[0][j] = x_test_energy_hourly[0][j + 1]
        x_test_energy_hourly[0][-1] = predict_value

    y_predict_hourly = np.asarray(y_predict_hourly)
    mse = mean_squared_error(y_test, y_predict_hourly)
    mape = mean_absolute_percentage_error(y_test, y_predict_hourly)
    r2 = r2_score(y_test, y_predict_hourly)
    print('MSE:', mse)
    print('MAPE:', mape)
    print('R2:', r2)

    if is_save:
        print('save result to file ...')
        with open(res_file_pref, 'a+') as f:
            f.write("lstm_dim:{}\ndense_weather_dim:{}\ndense_merge_dim:{}\ndropout_rate:{}\nlearning_rate:{}\n"
                    .format(model_config.LSTM_DIM, model_config.DENSE_WEATHER_DIM, model_config.DENSE_MERGE_DIM,
                            model_config.DROP_RATE, learning_rate))
            f.write("y_test:{}\n".format(','.join([str(v) for v in y_test])))
            f.write("y_predict:{}\n".format(','.join([str(v) for v in y_predict_hourly])))
            f.write("Deviation:{}\n".format(
                ','.join([str(abs(test - pred) / test) for test, pred in zip(y_test, y_predict_hourly)])))
            f.write("MSE:{}\nMAPE:{}\nR2:{}\n\n".format(mse, mape, r2))

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_predict_hourly, color='green')
        pyplot.show()

    return mse, mape'''


def load_model_from_file(index, res_file_pref):
    model = load_model(res_file_pref+"_"+index+"_model.h5")
    return model


def save_model_to_file(model, index, res_file_pref):
    '''save_info = {"model": model}
    with open(res_file_pref+"_"+index+"_model", 'wb') as f:
        pickle.dump(save_info, f, protocol=pickle.HIGHEST_PROTOCOL)'''
    model.save(res_file_pref+"_"+index+"_model.h5")


def run_saved_model(data, model_index, res_file_pref, is_draw=False):
    X_train_weather, X_train_energy, y_train, \
    X_valid_weather, X_valid_energy, y_valid, \
    X_test_weather, X_test_energy, y_test = data

    print('--- Load saved model ...')
    model = load_model_from_file(model_index, res_file_pref)

    print('--- Predict ...')
    y_pred = model.predict([X_test_energy, X_test_weather],
                           verbose=1)[:, 0]
    y_pred = np.exp(y_pred)-1

    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    #print ("------- size: ", len(y_test))
    #print ("------- size: ", len(y_pred))
    print('MSE:', mse)
    print('MAPE:', mape)
    print('R2:', r2)

    if is_draw:
        pyplot.plot(y_test)
        pyplot.plot(y_pred, color='red')
        pyplot.show()


def select_feature(data):
    x_train_weather, x_train_energy, y_train, \
    x_valid_weather, x_valid_energy, y_valid, \
    x_test_weather, x_test_energy, y_test = data

    # x_train_weather (550, 18), x_train_energy (550, 49, 1), y_train (550,)
    # x_valid_weather (44, 18),  x_valid_energy (44, 49, 1),  y_valid (44,)
    # x_test_weather  (48, 18),  x_test_energy  (48, 49, 1),  y_test  (48,)

    # -----------------
    # process data
    # -----------------
    # resize energy arrays into 2d array
    x_train_energy = np.reshape(x_train_energy, (550, 49))
    x_valid_energy = np.reshape(x_valid_energy, (44, 49))
    x_test_energy = np.reshape(x_test_energy, (48, 49))

    # concatenate weather and enery data (49 + 18)
    x_train = np.concatenate((x_train_weather, x_train_energy), axis=1)
    x_valid = np.concatenate((x_valid_weather, x_valid_energy), axis=1)
    x_test = np.concatenate((x_test_weather, x_test_energy), axis=1)
    print("x train/valid/test dimension: ", x_train.shape, x_valid.shape, x_test.shape)

    # concatenate train and valid data (550 + 44)
    x_train_valid = np.concatenate((x_train, x_valid))
    y_train_valid = np.concatenate((y_train, y_valid))
    print("x/y train_valid dimension: ", x_train_valid.shape, y_train_valid.shape)

    # -----------------
    # feature selection
    # -----------------
    feature_total_num = 12
    selected = feature_selection(x_train_valid, y_train_valid, feature_total_num)
    print('selected features (%d):' % sum(selected), selected)
    x_train_valid = x_train_valid[:, selected]
    x_test = x_test[:, selected]
    print('after feature selection, train_valid/test dimension: ', x_train_valid.shape, x_test.shape)

    # -----------------
    # splits into PowerNet format
    # -----------------
    # splits train and valid data
    x_train = x_train_valid[:550]
    x_valid = x_train_valid[550:]
    print('x train/valid dimension: ', x_train.shape, x_valid.shape)
    y_train = y_train_valid[:550]
    y_valid = y_train_valid[550:]
    print('y train/valid dimension: ', y_train.shape, y_valid.shape)

    # splits to weather and enery
    weather_para_num = 0
    for i in selected[:18]:
        if i:
            weather_para_num += 1
    print("number weather parameters selected: ", weather_para_num)
    x_train_weather = x_train[:, :weather_para_num]
    x_train_energy = x_train[:, weather_para_num:]
    x_valid_weather = x_valid[:, :weather_para_num]
    x_valid_energy = x_valid[:, weather_para_num:]
    x_test_weather = x_test[:, :weather_para_num]
    x_test_energy = x_test[:, weather_para_num:]
    print('x train weather/enery: ', x_train_weather.shape, x_train_energy.shape)
    print('x valid weather/enery: ', x_valid_weather.shape, x_valid_energy.shape)
    print('x test weather/enery: ', x_test_weather.shape, x_test_energy.shape)

    # resize energy to 3d array
    energy_para_num = feature_total_num - weather_para_num
    x_train_energy = np.reshape(x_train_energy, (550, energy_para_num, 1))
    x_valid_energy = np.reshape(x_valid_energy, (44, energy_para_num, 1))
    x_test_energy = np.reshape(x_test_energy, (48, energy_para_num, 1))
    print('train/valid/test energy dimension', x_train_energy.shape, x_valid_energy.shape, x_test_energy.shape)

    return x_train_weather, x_train_energy, y_train, \
           x_valid_weather, x_valid_energy, y_valid, \
           x_test_weather, x_test_energy, y_test


def main():
    """
     69 spring
     27 summer
     17 winter
     5 winter
     39 autumn
     30 autumn
    """
    # apt_name = 69
    ss = 'spring'
    freq = '1h'
    season = {
        'autumn': {'trb': '2016-02-01', 'tre': '2016-02-25',
                   'vab': '2016-02-26', 'vae': '2016-02-27',
                   'teb': '2016-02-28', 'tee': '2016-02-29'},
        'summer': {'trb': '2016-07-01', 'tre': '2016-07-25',
                   'vab': '2016-07-26', 'vae': '2016-07-27',
                   'teb': '2016-07-29', 'tee': '2016-07-30'},
        'spring': {'trb': '2016-04-01', 'tre': '2016-04-26',
                   'vab': '2016-04-27', 'vae': '2016-04-28',
                   'teb': '2016-04-29', 'tee': '2016-04-30'},
        'winter': {'trb': '2016-11-01', 'tre': '2016-11-28', 'teb': '2016-11-29', 'tee': '2016-11-30'},
    }

    res_file_pref = '%sapt%d_%s' % (LSTM_RES_DIR % freq, 0, ss)

    
    # Load saved model and run
    load_saved = 0
    if load_saved:
        apt_name = 0
        res_file_pref = '%sapt%d_%s' % (LSTM_RES_DIR % freq, apt_name, ss)
        draw_bool = True

        data = load_apt_data(apt_name=APT_CSV % 0, data_split=season[ss], freq=freq)
        model_index = "4468"
        run_saved_model(data, model_index, res_file_pref=res_file_pref, is_draw=draw_bool)

        return

    
    # Tuning loop
    '''dim_parameters = [128]
    dense_weather_dim_parameters = [64]
    dense_merge_dim_parameters = [512]
    dropout_rates = [0.2]
    learning_rates = [0.001]
    '''
    appartments = [5, 17, 27, 30, 39, 69]
    dim_parameters = []
    for v in range(32, 1025, 32):
        dim_parameters.append(v)
    dense_weather_dim_parameters = [32, 64, 128, 256, 512, 1024]
    dense_merge_dim_parameters = [32, 64, 128, 256, 512, 1024]
    dropout_rates = [0.2]
    learning_rates = [0.01]
    draw_bool = False
    write_bool = True

    min_mse = {"lstm": 0, "dense_weather_dim": 0, "dense_merge_dim": 0, "dropout_rate": 0, "learning_rate": 0,
               "mse": 1000, "mape": 1000, "index": 0}
    min_mape = {"lstm": 0, "dense_weather_dim": 0, "dense_merge_dim": 0, "dropout_rate": 0, "learning_rate": 0,
                "mse": 1000, "mape": 1000, "index": 0}

    if not os.path.isdir(LSTM_RES_DIR % freq):
        print("The directory {} doesn't exist. Please either create a new one or change the default path in setting,py"
              .format(LSTM_RES_DIR % freq))
        return

    for apt_name in [0]:
        res_file_pref = '%sapt%d_%s' % (LSTM_RES_DIR % freq, apt_name, ss)
        start_time = time.time()

        # reset min for mse and mape
        min_mse = {x: 1000 for x in min_mse}
        min_mape = {x: 1000 for x in min_mape}

        data = load_apt_data(apt_name=APT_CSV % apt_name, data_split=season[ss], freq=freq)
        data = select_feature(data)

        set_index = 0
        for lstm_dim in dim_parameters:
            for dense_weather_dim in dense_weather_dim_parameters:
                for dense_merge_dim in dense_merge_dim_parameters:
                    for dropout_rate in dropout_rates:
                        PowerNetConfig.LSTM_DIM = lstm_dim
                        PowerNetConfig.DENSE_WEATHER_DIM = dense_weather_dim
                        PowerNetConfig.DENSE_MERGE_DIM = dense_merge_dim
                        PowerNetConfig.DROP_RATE = dropout_rate
                        for learning_rate in learning_rates:
                            try:
                                mse, mape = run_model(data,
                                                      res_file_pref=res_file_pref,
                                                      model_config=PowerNetConfig,
                                                      is_draw=draw_bool,
                                                      is_save=write_bool,
                                                      learning_rate=learning_rate,
                                                      set_index=set_index)
                                set_index += 1
                            except Exception as e:
                                with open('%sapt%d_%s' % (LSTM_RES_DIR % freq, apt_name, ss), 'a+') as f:
                                    print("Traceback while running: \n, ", str(e))
                                    f.write("\n\nTraceback while running:\n" + str(e) + "\n\n")
                                set_index += 1
                                continue

                            if mse < min_mse['mse']:
                                min_mse['mse'] = mse
                                min_mse['mape'] = mape
                                min_mse['lstm'] = lstm_dim
                                min_mse['dense_weather_dim'] = dense_weather_dim
                                min_mse['dense_merge_dim'] = dense_merge_dim
                                min_mse['dropout_rate'] = dropout_rate
                                min_mse['learning_rate'] = learning_rate
                                min_mse['index'] = set_index-1
                                #save_model_to_file(model, "mse", res_file_pref=res_file_pref)
                            if mape < min_mape['mape']:
                                min_mape['mse'] = mse
                                min_mape['mape'] = mape
                                min_mape['lstm'] = lstm_dim
                                min_mape['dense_weather_dim'] = dense_weather_dim
                                min_mape['dense_merge_dim'] = dense_merge_dim
                                min_mape['dropout_rate'] = dropout_rate
                                min_mape['learning_rate'] = learning_rate
                                min_mape['index'] = set_index-1
                                #save_model_to_file(model, "mape", res_file_pref=res_file_pref)

        with open(res_file_pref, 'a+') as f:
            f.write("Min mse:{} with mape {} with index {}\n".format(min_mse['mse'], min_mse['mape'], min_mse['index']))
            f.write("lstm:{}\ndense_weather_dim:{}\ndense_merge_dim:{}\ndropout_rate:{}\nlearning_rate:{}\n\n"
                    .format(min_mse['lstm'], min_mse['dense_weather_dim'], min_mse['dense_merge_dim'],
                            min_mse['dropout_rate'], min_mse['learning_rate']))
            f.write("Min mape:{} with mse {} with index {}\n".format(min_mape['mape'], min_mape['mse'], min_mape['index']))
            f.write("lstm:{}\ndense_weather_dim:{}\ndense_merge_dim:{}\ndropout_rate:{}\nlearning_rate:{}\n\n"
                    .format(min_mape['lstm'], min_mape['dense_weather_dim'], min_mape['dense_merge_dim'],
                            min_mape['dropout_rate'], min_mape['learning_rate']))
            f.write("----Tuning APT %d costs %d seconds\n\n" % (apt_name, time.time() - start_time))


if __name__ == '__main__':
    main()
