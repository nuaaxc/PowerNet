import numpy as np
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, LSTM, BatchNormalization, concatenate

from models.layers import (EnergyEncodingLayer,
                           WeatherEncodingLayer,
                           PredictLayer,
                           AttentionWeight,
                           Attention,
                           LayerNormalization)


def build_model(model_config, energy_dim, weather_dim):
    lstm_dim = model_config['LSTM_DIM']
    denseW_dim = model_config['DENSE_WEATHER_DIM']
    denseA_dim = model_config['DENSE_ATTENTION_DIM']
    denseP_dim = model_config['DENSE_PREDICTION_DIM']
    drop_rate = model_config['DROP_RATE']
    lr = model_config['LR']

    # Energy part
    energy = Input(shape=(energy_dim, 1,), dtype='float32', name='energy_input')
    energy_encoding = EnergyEncodingLayer(lstm_dim, drop_rate)(energy)

    attention_weight = AttentionWeight(n_factor=5, hidden_d=denseA_dim)(energy_encoding)
    energy_encoding_attended = Attention()([attention_weight, energy_encoding])

    # Weather part
    weather = Input(shape=(weather_dim,), dtype='float32', name='weather_input')
    weather_encoding = WeatherEncodingLayer(denseW_dim,
                                            input_dim=K.int_shape(weather)[-1],
                                            dropout=drop_rate)(weather)

    # merge layer
    m = concatenate([energy_encoding_attended, weather_encoding])

    m = LayerNormalization()(m)

    # prediction layer
    prediction = PredictLayer(denseP_dim, input_dim=K.int_shape(m)[-1], dropout=drop_rate)(m)

    # model
    model = Model(inputs=[energy, weather],
                  outputs=prediction)

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.adam(lr=lr))

    # model.summary()

    return model
