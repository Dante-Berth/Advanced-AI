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


class MetaPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides):
        super(MetaPoolingLayer, self).__init__()
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


class CNN_Layer(tf.keras.layers.Layer):

    def __init__(self, filters: int, kernel_size: int, activation: tf.keras.activations, pool_size: int, strides: int, pooling_layer_name: str):
        super(CNN_Layer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.pooling_layer_name = pooling_layer_name
        self.cn_layer = None
        self.pooling_layer= None
        print("init")


    def build(self, input_shape):
        if self.pooling_layer_name == "MetaPoolingLayer":
            self.pooling_layer = MetaPoolingLayer(self.pool_size, self.strides)
        else:
            if len(input_shape) == 3 and self.pooling_layer_name in ["AveragePooling2D", "MaxPooling2D"]:
                if self.pooling_layer_name == "AveragePooling2D":
                    self.pooling_layer_name = "AveragePooling1D"
                else:
                    self.pooling_layer_name = "MaxPooling1D"
            if len(input_shape) == 4 and self.pooling_layer_name in ["AveragePooling1D", "MaxPooling1D"]:
                if self.pooling_layer_name == "AveragePooling1D":
                    self.pooling_layer_name = "AveragePooling2D"
                else:
                    self.pooling_layer_name = "MaxPooling2D"

            self.pooling_layer = getattr(tensorflow.keras.layers, self.pooling_layer_name)(self.pool_size, self.strides,
                                                                                           padding="same",
                                                                                           data_format="channels_last")

        if len(input_shape) == 3 or self.pooling_layer_name in ["AveragePooling1D", "MaxPooling1D"] and tf.shape(input) == 4:
            self.cn_layer = tensorflow.keras.layers.Conv1D(filters=self.filters,
                                                           kernel_size=self.kernel_size,
                                                           strides=self.strides,
                                                           padding="same",
                                                           data_format="channels_last")
        else:
            self.cn_layer = tensorflow.keras.layers.Conv2D(filters=self.filters,
                                                           kernel_size=self.kernel_size,
                                                           strides=self.strides,
                                                           padding="same",
                                                           data_format="channels_last")
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, input, **kwargs):
        x = self.cn_layer(input)
        x = self.activation(x)
        x = self.pooling_layer(x)
        x = self.batch_norm(x)
        return x

tensor_3 = tf.ones((12,24,36))
tensor_4 = tf.ones((12,24,36,48))
a = CNN_Layer(filters=3, kernel_size=2, activation=tf.keras.activations.gelu, pool_size=3, strides=3, pooling_layer_name="AveragePooling1D")
print(a(tensor_3))
print(CNN_Layer(3, 2, tf.keras.activations.gelu, 3, 3, "AveragePooling1D")(tensor_4))
print(CNN_Layer(3, 2, tf.keras.activations.gelu, 3, 3, "MetaPoolingLayer")(tensor_4))


