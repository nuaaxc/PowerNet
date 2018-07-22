import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, BatchNormalization, concatenate

from models.layers import EnergyEncodingLayer, WeatherEncodingLayer, PredictLayer, AttentionWeight, Attention


def build_model(model_config, energy_dim, weather_dim):
    lstm_dim = model_config['LSTM_DIM']
    denseA_dim = model_config['DENSE_ATTENTION_DIM']
    denseP_dim = model_config['DENSE_PREDICTION_DIM']
    drop_rate = model_config['DROP_RATE']
    lr = model_config['LR']

    # Energy part
    energy = Input(shape=(energy_dim, 1,), dtype='float32', name='energy_input')
    energy_encoding = EnergyEncodingLayer(lstm_dim, drop_rate)(energy)

    attention_weight = AttentionWeight(n_factor=1, hidden_d=denseA_dim)(energy_encoding)
    energy_encoding = Attention()([attention_weight, energy_encoding])

    # Weather part
    weather = Input(shape=(weather_dim,), dtype='float32', name='weather_input')

    # prediction layer
    prediction = PredictLayer(denseP_dim,
                              input_dim=K.int_shape(energy_encoding)[-1],
                              dropout=drop_rate)(energy_encoding)

    # model
    model = Model(inputs=[energy, weather],
                  outputs=prediction)

    optimizer = keras.optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-05, schedule_decay=0.0)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)

    # model.summary()

    return model
