import tensorflow as tf
import tensorflow_probability as tfp
from Activation import CustomActivationLayers
tfd = tfp.distributions

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
    Layer faisant intervenir les fonctions linalg pour un tensor
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
    def __init__(self):
        super(LinalgMonolayer, self).__init__()
        self.hyperparameter = {
            "hyperparameter_linalg_mono_functions": ['adjoint', 'l2_normalize',  'matrix_transpose', 'normalize', 'qr', 'svd']
        }

    def get_hyperparameters(self):
        return self.hyperparameter

    def call(self, inputs,linalg_mono_functions="adjoint"):
        if linalg_mono_functions == "normalize" or linalg_mono_functions == "qr" or linalg_mono_functions == "svd":
            return getattr(tf.linalg, linalg_mono_functions)(inputs)[0]
        else:
            return getattr(tf.linalg, linalg_mono_functions)(inputs)




class SignalLayer(tf.keras.layers.Layer):
    """
    SignalLayer propose plusieurs transformations de fourrier.
    """
    def __init__(self):
        super(SignalLayer, self).__init__()
        self.hyperparameter = {
            "hyperparameter_signal": ["rfft", "rfft2d", "dct", "fft", "fft2d", "ifft", "ifft2d"]
        }

    def get_hyperparameters(self):
        return self.hyperparameter

    def call(self, inputs, signal_name):
        if signal_name in ["rfft", "rfft2d", "dct"]:
            return tf.cast(getattr(tf.signal, signal_name)(inputs), dtype=tf.dtypes.float32)
        else:
            return tf.cast(
                getattr(tf.signal, signal_name)((tf.cast(inputs, dtype=tf.dtypes.complex64))),
                dtype=tf.dtypes.float32,
            )





class CorLayer(tf.keras.layers.Layer):
    def __init__(self, study_network="Dense", activation="softmax", sample_axis=-1, event_axis=-1):
        """
        Fonction calculant la corrélation linéaire ou pas entre soit les marchés, soit les features selon des pas de temps de fixes ou bien la corrélation d'une suite temporelle avec une autre

        Input (batch,time_steps,features) (batch, time_steps, time_steps) if sample_axis=-1,event_axis=-2

        Input (batch,time_steps,features,markets) (batch, time_steps, markets,markets) if sample_axis=-1,event_axis=-2
        ; regarde la corrélation entre features pour un pas de temps fixé pour l'ensemble des marchés

        Input (batch,time_steps,features) (batch, features, features) if sample_axis=-2,event_axis=-1

        Input (batch,time_steps,features,markets) (batch, time_steps, markets,markets) if sample_axis=-2,event_axis=-1
        ; regarde la corrélation entre marchés pour un pas de temps fixé et pour l'ensemble des features

        Input (batch,time_steps,features,markets) (batch, time_steps, time_steps,features) if sample_axis=-1,event_axis=-3

        Input (batch,time_steps,features,markets) (batch, time_steps, time_steps,markets) if sample_axis=-2,event_axis=-3
        Parameters
        ----------
        study_network : str
        activation : str ou bien tf.activation
        """
        self.layer_a = None
        self.activation = CustomActivationLayers.MetaActivationLayer()
        self.study_network = study_network
        self.sample_axis = sample_axis
        self.event_axis = event_axis
        self.hyperparameter = {
            "hyperparameter_list_layers": all_combination_list(["dense", "convolution", "time"]),
            "hyperparameter_sample_axis": [-3, -2, -1],
            "hyperparameter_event_axis": [-3, -2, -1],
        }
        super(CorLayer, self).__init__()

    def get_hyperparameters(self):
        return self.hyperparameter

    def build(self, input_shape):
        num_units = input_shape[-1]
        if "dense" in self.study_network:
            self.layer_a = tf.keras.layers.Dense(num_units, activation=self.activation)
        elif "convolution" in self.study_network:
            self.layer_a = tf.keras.layers.Conv1D(filters=num_units, kernel_size=1, padding="same",
                                                  activation=self.activation)
        if "time" in self.study_network and self.layer_a:
            self.layer_a = tf.keras.layers.TimeDistributed(self.layer_a)

    def call(self, inputs):
        if self.layer_a:
            output = self.layer_a(inputs)
        else:
            output = inputs

        if len(inputs.shape) == 3 and self.event_axis == -3 or self.event_axis == self.sample_axis:
            event_axis = -2
            sample_axis = -1

        return tfp.stats.correlation(output, sample_axis=sample_axis, event_axis=event_axis)


class MetaPoolinglayer(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides):
        super(MetaPoolinglayer, self).__init__()
        self.pool_size = pool_size
        self.strides = strides
        self.hyperparameter = {
            "pool_size": [2, 3, 4, 5, 6],
            "strides": [1, 2, 3, 4, 5, 6]
        }
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

    def get_hyperparameters(self):
        return self.hyperparameter

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
