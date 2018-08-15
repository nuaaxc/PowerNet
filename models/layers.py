from keras.models import Sequential
from keras.layers import merge, concatenate, add
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout, Lambda
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.engine.topology import Layer
from keras.layers import TimeDistributed
import keras.backend as K
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import keras


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=keras.initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class EnergyEncodingLayer(object):
    def __init__(self, dim, drop_rate):
        # self.lstm1 = GRU(dim,
        #                  return_sequences=True,
                         # kernel_regularizer=regularizers.l2(0.0001),
                         # dropout=drop_rate)
        self.lstm2 = GRU(dim,
                         return_sequences=True,
                         # kernel_regularizer=regularizers.l2(0.0001),
                         )
        self.batch_norm = BatchNormalization()

    def __call__(self, x):
        # x = self.lstm1(x)
        x = self.lstm2(x)
        # x = self.batch_norm(x)
        return x


class WeatherEncodingLayer(object):
    def __init__(self, dim, input_dim=0, dropout=0.1):
        self.model = Sequential()
        self.model.add(Dense(dim, activation='relu', input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        # self.model.add(Dense(dim, activation='relu'))
        # self.model.add(Dropout(dropout))

    def __call__(self, x):
        return self.model(x)


class AttentionWeight(Layer):
    def __init__(self, n_factor=5, hidden_d=32, **kwargs):
        self.n_factor = n_factor
        self.hidden_d = hidden_d
        super(AttentionWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        embedding_size = input_shape[-1]
        sequence_length = input_shape[1]
        self.W1 = self.add_weight(shape=(embedding_size, self.hidden_d),
                                  name='W1',
                                  initializer='glorot_uniform',
                                  # regularizer=regularizers.l2(0.01),
                                  )
        # self.W2 = self.add_weight(shape=(self.hidden_d, self.n_factor),
        #                           name='W2',
        #                           initializer='glorot_uniform',
        #                           # regularizer=regularizers.l2(0.01),
        #                           )
        super(AttentionWeight, self).build(input_shape)

    def call(self, x, **kwargs):
        # weight = K.softmax(K.dot(K.relu(K.dot(x, self.W1)), self.W2))
        weight = K.softmax(K.dot(x, self.W1))
        return K.permute_dimensions(weight, (0, 2, 1))  # batch_size, n_factor, sequence_length

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape[0], self.n_factor, input_shape[1]

    def get_config(self):
        config = {'n_reason': self.n_factor,
                  'hidden_d': self.hidden_d}
        base_config = super(AttentionWeight, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.n_factor = input_shape[0][1]
        self.seq_length = input_shape[0][2]
        self.hidden_size = input_shape[1][2]
        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        W, H = inputs
        E = K.batch_dot(W, H)
        E = K.mean(E, axis=1)
        return E

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.hidden_size


class PredictLayer(object):
    def __init__(self, dim, input_dim=0, dropout=0.1):
        self.model = Sequential()
        self.model.add(Dense(dim, activation='relu', input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        # self.model.add(Dense(dim, activation='relu'))
        # self.model.add(Dropout(dropout))
        self.model.add(Dense(1))

    def __call__(self, x):
        return self.model(x)
