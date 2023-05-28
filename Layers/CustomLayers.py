import tensorflow as tf
from Activation import CustomActivationLayers
from Fromtwotensorsintoonetensor import R_ListTensor


def all_combination_list(input_liste):
    """
    fonction listant toutes les possibilités et séparant chaque item par "_".

    e.g. :

    input_liste = ["apple","orange]

    all_combination_list(input_liste) = ["apple","orange","apple_orange"]
    Parameters
    ----------
    input_liste: list[str]

    Returns: list
    -------

    """
    all_combinations = []
    # Append all combinations of those combinations
    for r in range(1, len(input_liste) + 1):
        for combination in input_liste.combinations(input_liste, r):
            all_combinations.append("_".join(combination))
    return all_combinations



class LinalgMonolayer(tf.keras.layers.Layer):
    """
    tensor_shape_four = tf.random.uniform((32, 24, 6, 4))
    tensor_shape_three = tf.random.uniform((32, 24, 6))
    adjoint
    (32, 6, 24)
    adjoint
    (32, 24, 4, 6)
    l2_normalize
    (32, 24, 6)
    l2_normalize
    (32, 24, 6, 4)
    matrix_transpose
    (32, 6, 24)
    matrix_transpose
    (32, 24, 4, 6)
    normalize
    (32, 24, 6)
    normalize
    (32, 24, 6, 4)
    qr
    (32, 24, 6)
    qr
    (32, 24, 6, 4)
    svd
    (32, 6)
    svd
    (32, 24, 4)
    """
    def __init__(self,linalg_mono_functions):
        self.linalg_mono_functions = linalg_mono_functions
        self.linalg_layer = getattr(tf.linalg, self.linalg_mono_functions)
        super(LinalgMonolayer, self).__init__()

    @staticmethod
    def get_name():
        return "linalg_layer"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_linalg_mono_functions": ['adjoint', 'l2_normalize',  'matrix_transpose', 'normalize', 'qr', 'svd']
        }
    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = R_ListTensor()



    def call(self, input):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)
        if self.linalg_mono_functions in ["normalize","qr","svd"]:
            return self.linalg_layer(input)[0]
        else:
            return self.linalg_layer(input)




class SignalLayer(tf.keras.layers.Layer):
    """
    SignalLayer proposes several Fourrier transforms.
    """
    def __init__(self,signal_name):
        self.signal_name = signal_name
        self.signal_layer = getattr(tf.signal, self.signal_name)
        super(SignalLayer, self).__init__()

    @staticmethod
    def get_name():
        return "signal_layer"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_signal": ["rfft", "rfft2d", "dct", "fft", "fft2d", "ifft", "ifft2d"]
        }
    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = R_ListTensor()



    def call(self, input):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)
        if self.signal_name in ["rfft", "rfft2d", "dct"]:
            return tf.cast(self.signal_layer(input), dtype=tf.dtypes.float32)
        else:
            return tf.cast(
                self.signal_layer((tf.cast(input, dtype=tf.dtypes.complex64))),
                dtype=tf.dtypes.float32,
            )



if __name__ == "__main__":
    tensor_3 = tf.ones((12, 24, 36))
    tensor_4 = tf.ones((12, 24, 36, 48))



    linalgmonolayer = LinalgMonolayer("adjoint")

    # Pass the input tensor through the layer
    tensor_5 = [tensor_4, tensor_3]
    output = linalgmonolayer(tensor_5)

    perceptron_layer = LinalgMonolayer("adjoint")
    output = perceptron_layer(tensor_4)
    print("success")
