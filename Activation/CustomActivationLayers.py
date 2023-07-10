import tensorflow as tf
"""
MetaActivationLayer généralise le concept de couche d'activation
"""



@tf.keras.utils.register_keras_serializable()
class Expcos(tf.keras.layers.Layer):
    """
    Activation function X -> sign(X)*\exp(w_{2}\cos(w_{1}X))
    """
    def __init__(self):
        super(Expcos, self).__init__()
        self.weight_1 = self.add_weight(
            name='weight_1',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True
        )
        self.weight_2 = self.add_weight(
            name='weight_2',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-3.1415, maxval=3.1415),
            trainable=True
        )
    def call(self, inputs):
        return tf.math.sign(inputs)*tf.math.exp(self.weight_2*tf.math.cos(self.weight_1*inputs))
@tf.keras.utils.register_keras_serializable()
class Signlog(tf.keras.layers.Layer):
    """
    Activation function from a paper Dreamer but adding a weight for increasing or reducing the input importance
    """
    def __init__(self):
        super(Signlog, self).__init__()
        self.weight = self.add_weight(
            name='weights',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-10, maxval=10),
            trainable=True
        )

    def call(self, inputs):
        return tf.math.sign(inputs) * tf.math.log(tf.keras.activations.relu(self.weight)*tf.math.abs(inputs) + 1)


@tf.keras.utils.register_keras_serializable()
class MetaActivationLayer(tf.keras.layers.Layer):
    """
    Idea taken from AutoML Springer P.66
    """
    def __init__(self, **kwargs):
        super(MetaActivationLayer, self).__init__(**kwargs)
        self.activation_list = [
            tf.keras.activations.gelu,
            tf.keras.activations.softmax,
            tf.keras.activations.softsign,
            Signlog(),
            Expcos()
        ]
        self.num_weights = len(self.activation_list)
        self.weights_list = self.add_weight(
            name='weights',
            shape=(self.num_weights, 1),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True
        )
    def get_weights(self):
        return self.weights_list

    def call(self, inputs):
        weighted_activations = tf.zeros_like(inputs)
        for weight, activation in zip(tf.unstack(self.weights_list, axis=0), self.activation_list):
            weighted_activations += weight * activation(inputs)
        return weighted_activations
if __name__=="__main__":
    tensor_123 = tf.ones((4, 5, 6, 7))
    print(MetaActivationLayer()(tensor_123))















