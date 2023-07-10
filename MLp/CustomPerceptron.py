import tensorflow as tf
from Fromtwotensorsintoonetensor import R_ListTensor
from Activation.CustomActivationLayers import MetaActivationLayer
@tf.keras.utils.register_keras_serializable()
class Perceptron_Layer(tf.keras.layers.Layer):
    """
    Perceptron it is exactly like the layer from tensorflow however i added two stasticmethod for hyperoptimization
    """
    def __init__(self,units,activation):
        super(Perceptron_Layer, self).__init__()
        self.units = units
        if activation == "MetaActivationLayer":
            activation = MetaActivationLayer()
        self.activation = activation
        self.dense = tf.keras.layers.Dense(self.units,activation=self.activation)

    @staticmethod
    def get_name():
        return "dense"

    @staticmethod
    def get_layer_hyperparemeters():
        return {
            "hyperparameter_units": [8, 128],
            "hyperparameter_activation": ["gelu", "softsign", "softmax","MetaActivationLayer"]
        }
    def build(self, input_shape):
        if isinstance(input_shape,list):
            self.R_ListTensor = R_ListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)
        self.dense.build(input_shape)

    def call(self,input):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)
        return self.dense(input)


if __name__ == "__main__":
    tensor_3 = tf.ones((12, 24, 36))
    tensor_4 = tf.ones((12, 24, 36, 48))



    perceptron_layer = Perceptron_Layer(units=20,activation=tf.keras.activations.gelu)

    # Pass the input tensor through the layer
    tensor_5 = [tensor_4, tensor_3]
    output = perceptron_layer(tensor_5)

    perceptron_layer = Perceptron_Layer(units=20, activation=tf.keras.activations.gelu)
    output = perceptron_layer(tensor_4)

    perceptron_layer = Perceptron_Layer(units=20, activation="MetaActivationLayer")
    output = perceptron_layer([tensor_4,None])
    print("success")



