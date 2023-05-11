import tensorflow as tf


def transform_tensor(tensor: tf.Tensor):
    """
    transform a tensor of length shape >=3 into a tensor of shape equals to 3
    :param tensor: tf.Tensor is the tensor of length shape >3
    :return: a of tensors of shape (batch_size, sequence_length, embedding_size)
    """
    shape = tf.shape(tensor)
    batch_size = shape[0]
    sequence_length = shape[1]
    embedding_size = tf.reduce_prod(shape[2:])
    transformed_tensor = tf.reshape(tensor, (batch_size, sequence_length, embedding_size))

    return transformed_tensor


class RNN_Layer(tf.keras.layers.Layer):
    """
    RNN_layer generalizes the concept of RNN,  can handle input of different size either 3 or either 4
    """

    def __init__(self, rnn_layer, units: int, dropout: float,
                 recurrent_dropout: float, reduction_factor_input: int, reduction_factor_output: int,
                 bidirectionnal_layer: bool):
        """
        Initialize the RNN_Layer.

        :param rnn_layer: The specific RNN layer to use (e.g., tf.keras.layers.SimpleRNN, tf.keras.layers.LSTM).
        :param units: The number of units or neurons in the RNN layer.
        :param dropout: The dropout rate to apply to the RNN layer.
        :param recurrent_dropout: The recurrent dropout rate to apply to the RNN layer.
        :param reduction_factor_input: The reduction factor to apply to the input dimension.
        :param reduction_factor_output: The reduction factor to apply to the output dimension.
        :param bidirectionnal_layer:  A boolean indicating whether to use a bidirectional RNN layer or not.
        """
        super(RNN_Layer, self).__init__()
        self.rnn_layer = getattr(tf.keras.layers, rnn_layer)
        self.units = units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.reduction_factor_input = reduction_factor_input
        self.reduction_factor_output = reduction_factor_output
        self.bidirectionnal = bidirectionnal_layer
        self.reduction_layer_input = None
        self.reduction_layer_output = None

    @staticmethod
    def RNN_layer_hyperparemeters():
        return {
            "hyperparameter_rnn_layer": ["SimpleRNN", "LSTM", "GRU"],
            "hyperparameter_units": [2, 256],
            "hyperparameter_dropout": [0, 0.5, 0.1],
            "hyperparameter_recurrent_dropout": [0, 0.5, 0.1],
            "hyperparameter_reduction_factor_input": [1, 32],
            "hyperparameter_reduction_factor_output": [1, 16],
            "hyperparameter_bidirectionnal_layer": [True, False]
        }

    def build(self, input_shape):
        self.reduction_layer_input = tf.keras.layers.Conv1D(filters=input_shape[-1] // self.reduction_factor_input,
                                                            kernel_size=1)
        if self.bidirectionnal:
            self.rnn_layer = tf.keras.layers.Bidirectional(self.rnn_layer(units=self.units,
                                                                          return_sequences=True,
                                                                          dropout=self.dropout,
                                                                          recurrent_dropout=self.recurrent_dropout
                                                                          ))
        else:
            self.rnn_layer = self.rnn_layer(units=self.units,
                                            return_sequences=True,
                                            dropout=self.dropout,
                                            recurrent_dropout=self.recurrent_dropout
                                            )
        self.reduction_layer_output = tf.keras.layers.Conv1D(filters=self.units // self.reduction_factor_output,
                                                             kernel_size=1)

    def call(self, inputs, **kwargs):
        x = self.reduction_layer_input(inputs)
        x = self.rnn_layer(x)
        return self.reduction_layer_output(x)


if __name__ == "__main__":
    tensor_3 = tf.ones((12, 24, 36))
    tensor_4 = tf.ones((12, 24, 36, 48))
    print(tf.keras.layers.LSTM(22, return_sequences=True)(tensor_3))
    print(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(12, return_sequences=True))(tensor_3))
    print(RNN_Layer("LSTM",12,0.3,0.3,2,2,False)(tensor_3))
