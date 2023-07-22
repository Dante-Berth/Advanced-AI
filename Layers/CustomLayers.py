import tensorflow as tf
from Activation import CustomActivationLayers
from Fromtwotensorsintoonetensor import RListTensor


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


@tf.keras.utils.register_keras_serializable()
class LinalgMonolayer(tf.keras.layers.Layer):
    def __init__(self,linalg_mono_functions,**kwargs):
        self.linalg_mono_functions = linalg_mono_functions
        self.linalg_layer = getattr(tf.linalg, self.linalg_mono_functions)
        super(LinalgMonolayer, self).__init__(**kwargs)

    @staticmethod
    def get_name():
        return "linalg_layer"
    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_linalg_mono_functions": ['l2_normalize',  'matrix_transpose', 'normalize', 'qr']
        }
    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = RListTensor()



    def call(self, input):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)
        if self.linalg_mono_functions in ["normalize","qr","svd"]:
            output = self.linalg_layer(input)[0]
            if self.linalg_mono_functions == "svd":
                return tf.linalg.diag(output)
            else:
                return output
        else:
            return self.linalg_layer(input)
    def get_config(self):
        config = super(LinalgMonolayer, self).get_config()
        config.update({
            'linalg_mono_functions': self.linalg_mono_functions
        })
        return config


@tf.keras.utils.register_keras_serializable()
class SignalLayer(tf.keras.layers.Layer):
    """
    SignalLayer proposes several Fourrier transforms.
    """
    def __init__(self,signal_name,**kwargs):
        self.signal_name = signal_name
        self.signal_layer = getattr(tf.signal, self.signal_name)
        super(SignalLayer, self).__init__(**kwargs)

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
            self.R_ListTensor = RListTensor()

    def get_config(self):
        config = super(SignalLayer, self).get_config()
        config.update({
            'signal_name': self.signal_name
        })
        return config

    def call(self, input):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)
        if self.signal_name in ["rfft", "rfft2d", "dct"]:
            return tf.cast(self.signal_layer(input), dtype=tf.dtypes.float32)
        else:
            return tf.math.real(self.signal_layer((tf.cast(input, dtype=tf.dtypes.complex64))))




if __name__ == "__main__":
    vector_1 = tf.keras.layers.Input(shape=(5, 6, 2))
    signallayer = LinalgMonolayer("svd")
    ouputs = signallayer(vector_1)

    model = tf.keras.models.Model(inputs=vector_1, outputs=ouputs)
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
    vector_1 = tf.ones(shape=(12, 5, 6, 2))
    print(model.predict(vector_1))
    """
    tensor_3 = tf.ones((12, 24, 36))
    tensor_4 = tf.ones((12, 24, 36, 48))



    linalgmonolayer = LinalgMonolayer("adjoint")

    # Pass the input tensor through the layer
    tensor_5 = [tensor_4, tensor_3]
    output = linalgmonolayer(tensor_5)

    perceptron_layer = LinalgMonolayer("qr")
    output = perceptron_layer(tensor_4)
    print(tensor_4.shape,output.shape)

    print( tf.linalg.diag(tf.linalg.svd(
        tensor_4, full_matrices=True, compute_uv=True, name=None
    )[0]).shape)
    print(tf.linalg.svd(
        tensor_4, full_matrices=True, compute_uv=True, name=None
    )[1].shape)
    print(tf.linalg.svd(
        tensor_4, full_matrices=True, compute_uv=True, name=None
    )[2].shape)

    print("success")
    """
