import tensorflow as tf
from Fromtwotensorsintoonetensor import R_ListTensor
class MultiHeadAttention_Layer(tf.keras.layers.Layer):
    """
    MultiHeadAttention_Layer is the layer reffered to Multi Head Attention
    """

    def __init__(self, num_heads: int, key_dim: int, value_dim: int,dropout: float):
        super(self,MultiHeadAttention_Layer).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout = dropout

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.R_ListTensor = R_ListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)

    @staticmethod
    def get_name():
        return "multiheadattention"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_num_heads": [1, 10, 1],
            "hyperparameter_pool_size": [1, 10, 1],
            "hyperparameter_strides": [1, 10, 1],
            "hyperparameter_pooling_layer_name": ["MetaPoolingLayer", "AveragePooling2D", "MaxPooling2D",
                                                  "AveragePooling1D", "MaxPooling1D"]
        }

    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = R_ListTensor()
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
    tensor_3 = tf.ones((12,24,36))
    tensor_4 = tf.ones((12,24,36,48))
    a = CNN_Layer(filters=10, kernel_size=10, activation="gelu", pool_size=10, strides=10, pooling_layer_name="AveragePooling1D")
    print(a(tensor_3))
    print(CNN_Layer(40, 2, "gelu", 40, 40, "AveragePooling1D")(tensor_4))
    print(CNN_Layer(40, 2, "gelu", 3, 3, "MetaPoolingLayer")(tensor_4))
    print(CNN_Layer(3, 2, "gelu", 3, 3, "MetaPoolingLayer")([tensor_4,tensor_3]))
    print(CNN_Layer(3, 2, "gelu", 3, 3, "MetaPoolingLayer")([tensor_3, tensor_3]))