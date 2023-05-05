import tensorflow as tf

class Resnet(tf.keras.layers.Layer):
    """
    Resnet layer allows to know if the residual connexion is relevant, works only if the input_channels==output_channels for the major and minor neural network
    """

    def __init__(self, num_blocks: int = 2, major_neural_network=None,
                 final_activation_layer = tf.identity(), minor_neural_network=None):

        super().__init__()

        self.weight = self.add_weight(
            name='weight',
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=1),
            trainable=True
        )
        self.num_blocks = num_blocks
        self.major_neural_network = major_neural_network
        self.final_activation_layer = final_activation_layer
        self.minor_neural_network = minor_neural_network
        self.softmax_weights = None

    def call(self, data):
        """

        Args:
            data: it is a data
        Returns:
                x
        """
        self.weight = tf.keras.activations.sigmoid(self.weight)
        for i in range(self.num_blocks):
            data = self.major_neural_network(data)
            if self.minor_neural_network is None:
                data_x_ghost = tf.identity(data)
            else:
                data_x_ghost = self.minor_neural_network(data)
            data = self.final_activation_layer(self.weight*data + (1-self.weight)*data_x_ghost)

        return data
