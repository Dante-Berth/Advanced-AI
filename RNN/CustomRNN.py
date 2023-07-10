import tensorflow as tf
from Fromtwotensorsintoonetensor import R_ListTensor
@tf.keras.utils.register_keras_serializable()
class Reshape_Layer_3D(tf.keras.layers.Layer):
    def __init__(self):
        super(Reshape_Layer_3D, self).__init__()
        self.reshape = None

    def build(self, input_shape):
        if len(input_shape)>3:
            self.reshape = tf.keras.layers.Reshape((input_shape[1], tf.reduce_prod(input_shape[2:]).numpy()[()]))

    def call(self, inputs, *args, **kwargs):
        if self.reshape:
            return self.reshape(inputs)
        else:
            return inputs
@tf.keras.utils.register_keras_serializable()
class RNN_Layer(tf.keras.layers.Layer):
    """
    RNN_layer generalizes the concept of RNN,  can handle input of different size either 3 or either 4
    """

    def __init__(self, rnn_layer, units: int, dropout: float,
                 recurrent_dropout: float, reduction_factor_input: int, reduction_factor_output: int):
        """
        Initialize the RNN_Layer.

        :param rnn_layer: The specific RNN layer to use (e.g., tf.keras.layers.SimpleRNN, tf.keras.layers.LSTM).
        :param units: The number of units or neurons in the RNN layer.
        :param dropout: The dropout rate to apply to the RNN layer.
        :param recurrent_dropout: The recurrent dropout rate to apply to the RNN layer.
        :param reduction_factor_input: The reduction factor to apply to the input dimension.
        :param reduction_factor_output: The reduction factor to apply to the output dimension.
        """
        super(RNN_Layer, self).__init__()
        self.rnn_layer = getattr(tf.keras.layers, rnn_layer)
        self.units = units
        self.dropout = dropout//10*0.1
        self.recurrent_dropout = recurrent_dropout//10*0.1
        self.reduction_factor_input = reduction_factor_input
        self.reduction_factor_output = reduction_factor_output
        self.reduction_layer_input = None
        self.reduction_layer_output = None

    @staticmethod
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

    def build(self, input_shape):

        self.reduction_layer_input = tf.keras.layers.Conv1D(filters=tf.maximum(1,input_shape[-1] // self.reduction_factor_input),
                                                            kernel_size=1)


        self.rnn_layer = self.rnn_layer(units=self.units,
                                            return_sequences=True,
                                            dropout=self.dropout,
                                            recurrent_dropout=self.recurrent_dropout
                                            )
        self.reduction_layer_output = tf.keras.layers.Conv1D(filters=tf.maximum(1,self.units // self.reduction_factor_output),
                                                             kernel_size=1)

    def call(self, inputs):
        x = self.reduction_layer_input(inputs)
        x = self.rnn_layer(x)
        return self.reduction_layer_output(x)
@tf.keras.utils.register_keras_serializable()
class R_RNN_Layer(tf.keras.layers.Layer):
    def __init__(self, rnn_layer, units, dropout, recurrent_dropout,
                 reduction_factor_input, reduction_factor_output):
        super(R_RNN_Layer, self).__init__()
        self.reshape_layer = Reshape_Layer_3D()
        self.rnn_layer = RNN_Layer(rnn_layer, units, dropout, recurrent_dropout,
                                        reduction_factor_input, reduction_factor_output)

    @staticmethod
    def get_name():
        return "rnn"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_rnn_layer": ["SimpleRNN", "LSTM", "GRU"],
            "hyperparameter_units": [2, 256],
            "hyperparameter_dropout": [0, 50, 10],
            "hyperparameter_recurrent_dropout": [0, 50, 10],
            "hyperparameter_reduction_factor_input": [1, 32],
            "hyperparameter_reduction_factor_output": [1, 16]
        }
    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = R_ListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)
        self.reshape_layer.build(input_shape)
        reshaped_shape = self.reshape_layer.compute_output_shape(input_shape)
        self.rnn_layer.build(reshaped_shape)

    def call(self, inputs):
        if isinstance(inputs,list):
            inputs = self.R_ListTensor.call(inputs)

        x = self.reshape_layer.call(inputs)
        x = self.rnn_layer.call(x)
        return x



if __name__ == "__main__":
    tensor_3 = tf.ones((12, 24, 36))
    tensor_4 = tf.ones((12, 24, 36, 48))


    # Create an instance of ReshapeAndRNNLayer
    reshape_rnn_layer = R_RNN_Layer(rnn_layer="LSTM", units=12, dropout=0.3, recurrent_dropout=0.3,
                                           reduction_factor_input=2, reduction_factor_output=2)

    # Pass the input tensor through the layer
    tensor_5 = [tensor_4, tensor_3]
    output = reshape_rnn_layer(tensor_5)


    # Print the output shape
    print(output.shape)

    # Create an instance of ReshapeAndRNNLayer
    reshape_rnn_layer = R_RNN_Layer(rnn_layer="LSTM", units=12, dropout=0.3, recurrent_dropout=0.3,
                                           reduction_factor_input=2, reduction_factor_output=2)

    # Pass the input tensor through the layer
    output = reshape_rnn_layer(tensor_3)

    # Print the output shape
    print(output.shape)

