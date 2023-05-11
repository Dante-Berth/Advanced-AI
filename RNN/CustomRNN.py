import tensorflow as tf

class RNN_Layer(tf.keras.layers.Layer):
    """
    CNN_layer generalizes the concept of CNN,  can handle input of different size either 3 or either 4
    """

    def __init__(self, rnn_layer,units: int, return_sequences: bool, return_state: bool, dropout: float, recurrent_dropout: float):
        super(RNN_Layer, self).__init__()
        self.rnn_layer = getattr(tf.keras.layers, rnn_layer)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

    @staticmethod
    def RNN_layer_hyperparemeters():
        return {
            "hyperparameter_rnn_layer": ["SimpleRNN","LSTM","GRU"],
            "hyperparameter_units": [1, 64],
            "hyperparameter_return_sequences": [True, False],
            "hyperparameter_return_state": [True, False],
            "hyperparameter_dropout": [0,0.5,0.1],
            "hyperparameter_recurrent_dropout": [0, 0.5, 0.1]
        }

