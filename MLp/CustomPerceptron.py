import tensorflow as tf
class Perceptron_Layer(tf.keras.layers.Layer):
    """
    Perceptron it is exactly like the layer from tensorflow however i added two stasticmethod for hyperoptimization
    """
    def __init__(self,units,activation):
        super(Perceptron_Layer, self).__init__()
        self.units = units
        self.activation = activation
        self.dense = tf.keras.layers.Dense(self.units,self.activation)

    @staticmethod
    def get_name():
        return "dense"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_units": [8, 256],
            "hyperparameter_activation": ["gelu", "softsign", "softmax"]
        }
    def call(self,input):
        return self.dense(input)


