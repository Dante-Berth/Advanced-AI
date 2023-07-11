import tensorflow as tf
from Fromtwotensorsintoonetensor import R_ListTensor
from CNN.CustomCNN import MetaPoolingLayer

@tf.keras.utils.register_keras_serializable()
class ReductionLayerSVD(tf.keras.layers.Layer):
    def __init__(self, r):
        super(ReductionLayerSVD, self).__init__()
        self.r = r

    @staticmethod
    def get_name():
        return "reduction_svd"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_r": [1, 10, 1]
        }

    @staticmethod
    def rank_r_approx(s, U, V, r):

        s_r, U_r, V_r = s[..., :r], U[..., :, :r], V[..., :, :r]
        A_r = tf.einsum('...s,...us,...vs->...uv', s_r, U_r, V_r)
        return A_r

    def call(self, inputs):
        if isinstance(inputs, list):
            inputs = R_ListTensor()(inputs)
        s, U, V = tf.linalg.svd(inputs)
        return self.rank_r_approx(s, U, V, self.r)

@tf.keras.utils.register_keras_serializable()
class ReductionLayerPooling(tf.keras.layers.Layer):
    def __init__(self, ratio_pool_size: int, ratio_strides: int, ratio_dense: int, pooling_layer_name: str):
        super(ReductionLayerPooling, self).__init__()
        self.ratio_pool_size = ratio_pool_size
        self.ratio_strides = ratio_strides
        self.ratio_dense = ratio_dense
        self.pooling_layer_name = pooling_layer_name
        self.pooling_layer = None

    @staticmethod
    def get_name():
        return "reduction_layer_pooling"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_ratio_pool_size": [1, 3, 1],
            "hyperparameter_ratio_strides": [1, 3, 1],
            "hyperparameter_ratio_dense": [1, 7, 1],
            "hyperparameter_pooling_layer_name": ["MetaPoolingLayer", "AveragePooling2D", "MaxPooling2D",
                                                  "AveragePooling1D", "MaxPooling1D"]
        }

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.R_ListTensor = R_ListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)

        self.pool_size = max(min(max(input_shape[1] * self.ratio_pool_size // 10,1),input_shape[-2]-1),1)
        self.strides = max(input_shape[2] * self.ratio_strides // 10,1)

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
                                                                                   padding="valid",
                                                                                   data_format="channels_last")

        output_shape_pooling_layer = self.pooling_layer.compute_output_shape(input_shape)
        self.dense_layer = tf.keras.layers.Dense(units=max(input_shape[-1] * self.ratio_dense // 10,1))
        self.dense_layer.build(output_shape_pooling_layer)

    def call(self, input):
        if isinstance(input, list):
            input = self.R_ListTensor.call(input)
        x = self.pooling_layer(input)
        return self.dense_layer(x)


if __name__ == "__main__":
    tensor_3 = tf.random.uniform((12, 42, 8))
    tensor_4 = tf.random.uniform((12, 28, 8, 28))
    linalgmonolayer = ReductionLayerSVD(2)

    # Pass the input tensor through the layer
    output = linalgmonolayer(tensor_4)
    output = linalgmonolayer([tensor_3, tensor_4])
    #print(ReductionLayerPooling(5, 5, 3, "MetaPoolingLayer")([tensor_3, tensor_3]).shape)
    print("##############################################")
    print(ReductionLayerPooling(4, 4, 4, "AveragePooling2D")(tensor_3).shape)
    #print(ReductionLayerPooling(5, 6, 2, "MetaPoolingLayer")([tensor_3, tensor_4]).shape)


