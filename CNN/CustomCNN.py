import tensorflow as tf
from Fromtwotensorsintoonetensor import RListTensor
@tf.keras.utils.register_keras_serializable()
class MetaPoolingLayer(tf.keras.layers.Layer):
    """
    MetaPoolingLayer generalizes the concept of the pooling layer, can handle input of different size either 3 or either 4
    """
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

@tf.keras.utils.register_keras_serializable()
class CNN_Layer(tf.keras.layers.Layer):
    """
    CNN_layer generalizes the concept of CNN,  can handle input of different size either 3 or either 4
    """

    def __init__(self, filters: int, kernel_size: int, activation: tf.keras.activations, pool_size: int, strides: int, pooling_layer_name: str,**kwargs):
        """
        Initializes a CNN_Layer object.

        Args:
            filters (int): The number of filters in the convolutional layer.
            kernel_size (int): The size of the convolutional kernel.
            activation (tf.keras.activations): The activation function to use.
            pool_size (int): The size of the pooling window.
            strides (int): The stride size for the pooling operation.
            pooling_layer_name (str): The name of the pooling layer.

        Returns:
            None

        Raises:
            None
        """
        super(CNN_Layer, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = getattr(tf.keras.activations, activation)
        self.pool_size = pool_size
        self.strides = strides
        self.pooling_layer_name = pooling_layer_name
        self.cn_layer = None
        self.pooling_layer= None

    @staticmethod
    def get_name():
        return "cnn"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_filters": [8, 64],
            "hyperparameter_kernel_size": [1, 10, 1],
            "hyperparameter_activation": ["gelu", "softsign", "softmax", "tanh"],
            "hyperparameter_pool_size": [1, 10, 1],
            "hyperparameter_strides": [1, 10, 1],
            "hyperparameter_pooling_layer_name": ["MetaPoolingLayer", "AveragePooling2D", "MaxPooling2D",
                                                  "AveragePooling1D", "MaxPooling1D"]
        }

    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = RListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)

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

            self.pooling_layer = getattr(tf.keras.layers, self.pooling_layer_name)(self.pool_size, self.strides,
                                                                                           padding="same",
                                                                                           data_format="channels_last")

        if len(input_shape) == 3 or self.pooling_layer_name in ["AveragePooling1D", "MaxPooling1D"] and tf.shape(input) == 4:
            self.cn_layer = tf.keras.layers.Conv1D(filters=self.filters,
                                                           kernel_size=self.kernel_size,
                                                           strides=self.strides,
                                                           padding="same",
                                                           data_format="channels_last")
        else:
            self.cn_layer = tf.keras.layers.Conv2D(filters=self.filters,
                                                           kernel_size=self.kernel_size,
                                                           strides=self.strides,
                                                           padding="same",
                                                           data_format="channels_last")
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
    def get_config(self):
        config = super(CNN_Layer, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'activation': tf.keras.activations.serialize(self.activation),
            'pool_size': self.pool_size,
            'strides': self.strides,
            'pooling_layer_name': self.pooling_layer_name,
        })
        return config

    def call(self, input, **kwargs):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)

        x = self.cn_layer(input)
        x = self.batch_norm_1(x)
        x = self.activation(x)
        x = self.pooling_layer(x)
        x = self.batch_norm_2(x)
        return x
if __name__=="__main__":
    vector_1 = tf.keras.layers.Input(shape=(12, 5, 6, 2))
    vector_2 = tf.keras.layers.Input(shape=(10, 8, 14))
    a = CNN_Layer(filters=10, kernel_size=10, activation="gelu", pool_size=10, strides=10,
                  pooling_layer_name="AveragePooling1D")
    ouputs = a(vector_2)

    model = tf.keras.models.Model(inputs=vector_2, outputs=ouputs)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Model before the loading")
    model.summary()

    PATH = 'testing_model_custom_activation_layer.h5'
    model.save(PATH)
    model = tf.keras.models.load_model(PATH)
    print("Model loaded")
    print(model.summary())

    """
    tensor_3 = tf.ones((12,24,36))
    tensor_4 = tf.ones((12,24,36,48))
    a = CNN_Layer(filters=10, kernel_size=10, activation="gelu", pool_size=10, strides=10, pooling_layer_name="AveragePooling1D")
    print(a(tensor_3))
    print(CNN_Layer(40, 2, "gelu", 40, 40, "AveragePooling1D")(tensor_4))
    print(CNN_Layer(40, 2, "gelu", 3, 3, "MetaPoolingLayer")(tensor_4))
    print(CNN_Layer(3, 2, "gelu", 3, 3, "MetaPoolingLayer")([tensor_4,tensor_3]))
    print(CNN_Layer(3, 2, "gelu", 3, 3, "MetaPoolingLayer")([tensor_3, tensor_3]))
    """


