import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.regularizers import l2
import tensorflow as tf
from keras import regularizers
from keras.activations import *
from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization, Input, Conv1D, \
    AveragePooling1D, Concatenate, LSTM, SimpleRNN, LSTM, Bidirectional, TimeDistributed, ConvLSTM2D, Concatenate, \
    Dropout
from tensorflow.keras.models import Model
import optuna


class MetaPoolinglayer(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides):
        super(MetaPoolinglayer, self).__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.weight_average = self.add_weight(
            name='weight_average',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True
        )
        self.weight_max = self.add_weight(
            name='weight_max',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True
        )

    def get_weights(self):
        return self.weight_average, self.weight_max

    def build(self, input_shape):
        if len(input_shape) == 4:
            self.average_pooling_layer = tf.keras.layers.AveragePooling2D(self.pool_size,
                                                                          self.strides,
                                                                          padding="same",
                                                                          data_format="channels_last")
            self.max_pooling_layer = tf.keras.layers.MaxPool2D(self.pool_size,
                                                               self.strides,
                                                               padding="same",
                                                               data_format='channels_last')
        else:
            self.average_pooling_layer = tf.keras.layers.AveragePooling1D(self.pool_size,
                                                                          self.strides,
                                                                          padding="same",
                                                                          data_format="channels_last")
            self.max_pooling_layer = tf.keras.layers.MaxPool1D(self.pool_size,
                                                               self.strides,
                                                               padding="same",
                                                               data_format="channels_last")

    def call(self, inputs):

        return self.weight_max * self.max_pooling_layer(
            inputs) + self.weight_average * self.average_pooling_layer(inputs)


class CNN_layer(tf.keras.layers.Layers):

    def __init__(self, filters, kernel_size, activation, pool_size, strides, pooling_layer_name):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.pooling_layer_name = pooling_layer_name

    def build(self, input_shape):
        if self.pooling_layer_name in ["AveragePooling1D", "MaxPooling1D", "AveragePooling2D", "MaxPooling2D","Conv1D", "Conv2D"]:
            if len(input_shape) == 3 and self.pooling_layer_name in ["AveragePooling2D", "MaxPooling2D"]:
                if self.pooling_layer_name == "AveragePooling2D":
                    self.pooling_layer_name = "AveragePooling1D"
                else:
                    self.pooling_layer_name = "MaxPooling1D"

        self.pooling_layer = getattr(tensorflow.keras.layers, self.pooling_layer_name)(self.pool_size, self.strides,
                                                                                       padding="same",
                                                                                       data_format="channels_last")

    def call(self, input):
        if self.pooling_layer_name in ["AveragePooling1D", "MaxPooling1D"] and tf.shape(input) == 4:
            a,b,c,d = tf.shape(input)
            input = tf.reshape(input,(a,b,c*d))

        pooling_layer = self.pooling_layer(input)



def CNN_POOLING_layer(image, filters, kernel_size, activation, pool_size, stride, pooling_layer):
    output = Conv1D(filters=filters,
                    kernel_size=kernel_size,
                    activation=activation, padding="same")(image)
    output = BatchNormalization()(output)
    if pooling_layer == "AveragePooling1D":
        pool_layer = tf.keras.layers.AveragePooling1D
    elif pooling_layer == "MaxPooling1D":
        pool_layer = tf.keras.layers.MaxPooling1D

    output = pool_layer(pool_size=pool_size, strides=stride,
                        padding='same')(output)
    return output


def loop_CNN_layer(first_input, n_layers, trial, name_layer):
    CNN_OUTPUT = [f'{name_layer}_CNN_layers_outputs_{i}' for i in range(0, n_layers + 1)]
    CNN_OUTPUT[0] = first_input
    i = 0
    while i < n_layers:
        filters = trial.suggest_int(f'{name_layer}_CNN_filters_{i + 1}', 4, 128, log=True)
        kernel_size = trial.suggest_int(f'{name_layer}_CNN_kernel_size_{i + 1}', 1, 3, step=1),
        activation = trial.suggest_categorical(f'{name_layer}_CNN_activation_{i + 1}', ['gelu', 'tanh', 'swish'])
        pooling_layer = trial.suggest_categorical(f'{name_layer}_CNN_pooling_layer_{i + 1}',
                                                  ['AveragePooling1D', 'MaxPooling1D'])
        pool_size = trial.suggest_int(f'{name_layer}_CNN_pool_size_{i + 1}', 1, 4, step=1)
        stride = trial.suggest_int(f'{name_layer}_CNN_stride_{i + 1}', 1, 4, step=1)
        CNN_OUTPUT[i + 1] = CNN_POOLING_layer(CNN_OUTPUT[i], filters, kernel_size, activation, pool_size, stride,
                                              pooling_layer)
        i = i + 1
        last_rank = i
    return CNN_OUTPUT
