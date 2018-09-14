import keras
import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import (Input, Dense, Dropout, TimeDistributed,
                          LSTM, BatchNormalization, GlobalMaxPooling1D, GlobalAveragePooling1D)


class PredictLayer(object):
    def __init__(self, input_d, dense_d, dropout):
        self.model = Sequential()
        self.model.add(Dense(dense_d, input_shape=(1, input_d,), activation='sigmoid'))
        # self.model.add(Dropout(dropout))
        # self.model.add(Dense(1, input_shape=(input_d,)))
        self.model.add(Dense(1))

    def __call__(self, x):
        return self.model(x)


class RNNLayer(object):
    def __init__(self, lstm1_d, lstm2_d, dense_d, dropout):
        self.lstm1 = LSTM(lstm1_d,
                          return_sequences=True,
                          # kernel_regularizer=regularizers.l2(0.0001),
                          dropout=dropout)
        self.lstm2 = LSTM(lstm2_d,
                          return_sequences=True,
                          # kernel_regularizer=regularizers.l2(0.0001),
                          )
        self.dense = Dense(dense_d, activation='relu')
        self.max_pooling = GlobalMaxPooling1D()
        self.average_pooling = GlobalAveragePooling1D()
        self.output = Dense(1)

    def __call__(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)
        # x = self.max_pooling(x)
        x = self.average_pooling(x)
        return x


def build_model(model_config, feat_dim):
    x = Input(shape=(1, feat_dim,), dtype='float32', name='feat_input')

    f = RNNLayer(lstm1_d=model_config['LSTM_1_DIM'],
                 lstm2_d=model_config['LSTM_2_DIM'],
                 dense_d=model_config['DENSE_DIM'],
                 dropout=model_config['DROP_RATE'])(x)

    y = PredictLayer(input_d=K.int_shape(f)[-1],
                     dense_d=model_config['DENSE_DIM'],
                     dropout=model_config['DROP_RATE'])(f)

    model = Model(inputs=x, outputs=y)

    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.nadam(lr=model_config['LR']))

    return model
