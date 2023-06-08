import tensorflow as tf
from Fromtwotensorsintoonetensor import R_ListTensor
from CNN.CustomCNN import MetaPoolingLayer


class ReductionLayerSVD(tf.keras.layers.Layer):
    def __init__(self, r):
        super(ReductionLayerSVD, self).__init__()
        self.r = r

    @staticmethod
    def rank_r_approx(s, U, V, r):
        s_r, U_r, V_r = s[..., :r], U[..., :, :r], V[..., :, :r]
        A_r = tf.einsum('...s,...us,...vs->...uv', s_r, U_r, V_r)
        return A_r

    def call(self, inputs):
        # Perform SVD on the input tensor
        s, U, V = tf.linalg.svd(inputs)
        return self.rank_r_approx(s, U, V, self.r)


class ReductionLayerPooling(tf.keras.layers.Layer):
    def __init__(self, ratio_pool_size: int, ratio_strides: int, ratio_dense: int, pooling_layer_name: str):
        super(ReductionLayerPooling, self).__init__()
        self.ratio_pool_size = ratio_pool_size
        self.ratio_strides = ratio_strides
        self.ratio_dense = ratio_dense
        self.pooling_layer_name = pooling_layer_name
        self.pooling_layer = None

    @staticmethod
    def get_layer_hyperparameters():
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

        self.pool_size = input_shape[1] * self.ratio_pool_size // 10
        self.strides = input_shape[2] * self.ratio_strides // 10

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

        self.dense_layer = tf.keras.layers.Dense(units=input_shape[-1] * self.ratio_dense // 10)
        self.dense_layer.build(output_shape_pooling_layer)

    def call(self, input):
        if isinstance(input, list):
            input = self.R_ListTensor.call(input)
        x = self.pooling_layer(input)
        return self.dense_layer(x)


if __name__ == "__main__":
    tensor_3 = tf.random.uniform((12, 24, 36))
    tensor_4 = tf.random.uniform((12, 24, 36, 48))
    linalgmonolayer = ReductionLayerSVD(50)

    # Pass the input tensor through the layer
    output = linalgmonolayer(tensor_4)

    print(ReductionLayerPooling(2, 2, 4, "AveragePooling1D")(tensor_4).shape)
    print(ReductionLayerPooling(10, 10, 3, "MetaPoolingLayer")([tensor_3, tensor_3]).shape)
    print(ReductionLayerPooling(5, 6, 2, "MetaPoolingLayer")([tensor_3, tensor_4]).shape)
