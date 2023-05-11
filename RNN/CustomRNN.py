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

    def __init__(self, rnn_layer,units: int, return_sequences: bool, return_state: bool, dropout: float, recurrent_dropout: float, reduction:int):
        super(RNN_Layer, self).__init__()
        self.rnn_layer = getattr(tf.keras.layers, rnn_layer)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.reduction = reduction

    @staticmethod
    def RNN_layer_hyperparemeters():
        return {
            "hyperparameter_rnn_layer": ["SimpleRNN","LSTM","GRU"],
            "hyperparameter_units": [1, 64],
            "hyperparameter_return_sequences": [True, False],
            "hyperparameter_return_state": [True, False],
            "hyperparameter_dropout": [0,0.5,0.1],
            "hyperparameter_recurrent_dropout": [0, 0.5, 0.1]
            "hyperparameter_reduction":[8,256]
        }
if __name__=="__main__":
    tensor_3 = tf.ones((12,24,36))
    tensor_4 = tf.ones((12,24,36,48))
    print(tf.keras.layers.Conv1D(22,1)(transform_tensor(tensor_4)))
