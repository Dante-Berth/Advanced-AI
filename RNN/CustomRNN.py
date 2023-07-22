import tensorflow as tf
from Fromtwotensorsintoonetensor import RListTensor
@tf.keras.utils.register_keras_serializable()
class Reshape_Layer_3D(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Reshape_Layer_3D, self).__init__(**kwargs)
        self.reshape = None

    def get_config(self):
        config = super(Reshape_Layer_3D, self).get_config()
        return config

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

    def __init__(self, rnn_layer_str,units: int, dropout: float,
                 recurrent_dropout: float, reduction_factor_input: int, reduction_factor_output: int,*args,**kwargs):
        """
        Initialize the RNN_Layer.
        :param units: The number of units or neurons in the RNN layer.
        :param dropout: The dropout rate to apply to the RNN layer.
        :param recurrent_dropout: The recurrent dropout rate to apply to the RNN layer.
        :param reduction_factor_input: The reduction factor to apply to the input dimension.
        :param reduction_factor_output: The reduction factor to apply to the output dimension.
        :param rnn_layer: The specific RNN layer to use (e.g., tf.keras.layers.SimpleRNN, tf.keras.layers.LSTM).
        """
        super(RNN_Layer, self).__init__(*args,**kwargs)
        self.rnn_layer_str = rnn_layer_str
        self.rnn_layer = getattr(tf.keras.layers, rnn_layer_str)
        self.units = units
        self.dropout = dropout//10*0.1
        self.recurrent_dropout = recurrent_dropout//10*0.1
        self.reduction_factor_input = reduction_factor_input
        self.reduction_factor_output = reduction_factor_output
        self.reshape_layer = Reshape_Layer_3D()

    def get_config(self):
        config = super(RNN_Layer, self).get_config()
        config.update({
            'rnn_layer_str': self.rnn_layer_str,
            'units': self.units,
            'dropout': self.dropout,
            'recurrent_dropout': self.recurrent_dropout,
            'reduction_factor_input': self.reduction_factor_input,
            'reduction_factor_output': self.reduction_factor_output
        })
        return config

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
            self.R_ListTensor = RListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)
        self.reshape_layer.build(input_shape)
        input_shape = self.reshape_layer.compute_output_shape(input_shape)

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
        if isinstance(inputs,list):
            inputs = self.R_ListTensor.call(inputs)
        x = self.reshape_layer.call(inputs)
        x = self.reduction_layer_input(x)
        x = self.rnn_layer(x)
        return self.reduction_layer_output(x)
@tf.keras.utils.register_keras_serializable()
class R_RNN_Layer(RNN_Layer):
    def __init__(self,*args,**kwargs):
        super(R_RNN_Layer, self).__init__(*args,**kwargs)
        self.reshape_layer = Reshape_Layer_3D()

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
            self.R_ListTensor = RListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)
        self.reshape_layer.build(input_shape)
        reshaped_shape = self.reshape_layer.compute_output_shape(input_shape)
        super(RNN_Layer, self).build(reshaped_shape)

    def call(self, inputs):
        if isinstance(inputs,list):
            inputs = self.R_ListTensor.call(inputs)
        x = self.reshape_layer.call(inputs)
        x = super(RNN_Layer, self).call(x)
        return x
    def get_config(self):
        config = super(R_RNN_Layer, self).get_config()
        return config



if __name__ == "__main__":
    tensor_3 = tf.ones((12, 24, 36))
    tensor_4 = tf.ones((12, 24, 36, 48))


    # Create an instance of ReshapeAndRNNLayer
    reshape_rnn_layer = RNN_Layer(rnn_layer_str="LSTM", units=12, dropout=0.3, recurrent_dropout=0.3,
                                           reduction_factor_input=2, reduction_factor_output=2)

    # Pass the input tensor through the layer
    tensor_5 = [tensor_4, tensor_3]
    output = reshape_rnn_layer(tensor_5)


    # Print the output shape
    print(output.shape)

    # Create an instance of ReshapeAndRNNLayer
    reshape_rnn_layer = RNN_Layer(rnn_layer_str="LSTM", units=12, dropout=0.3, recurrent_dropout=0.3,
                                           reduction_factor_input=2, reduction_factor_output=2)

    # Pass the input tensor through the layer
    output = reshape_rnn_layer(tensor_3)

    # Print the output shape
    print(output.shape)
    dictionnary = {"rnn_layer_str":"LSTM", "units":12, "dropout":0.3, "recurrent_dropout":0.3,
                                           "reduction_factor_input":2, "reduction_factor_output":2 }
    liste = ["LSTM", 12, 0.3, 0.3,2, 2]

    rnn_layer_test = RNN_Layer(*liste)
    vector_1 = tf.keras.layers.Input(shape=(5, 12, 2))
    vector_2 = tf.keras.layers.Input(shape=(5,12))
    vector_3 = [vector_1, vector_2]
    ouputs = rnn_layer_test(vector_3)

    model = tf.keras.models.Model(inputs=[vector_1,vector_2], outputs=ouputs)
    model.compile(
        optimizer="Adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("Model before the loading")
    model.summary()

    PATH = 'testing_model_custom_activation_layer.h5'
    model.save(PATH)
    model_2 = tf.keras.models.load_model(PATH)
    print("Model loaded")
    print(model_2.summary())
    vector_1 = tf.ones(shape=(12, 5, 12, 2))
    vector_2 = tf.ones(shape=(12, 5, 12))
    print(model_2.predict([vector_1,vector_2]))

