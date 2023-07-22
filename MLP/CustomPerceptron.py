import tensorflow as tf
from Fromtwotensorsintoonetensor import RListTensor
from Activation.CustomActivationLayers import MetaActivationLayer
@tf.keras.utils.register_keras_serializable()
class Perceptron_Layer(tf.keras.layers.Layer):
    """
    Perceptron it is exactly like the layer from tensorflow however i added two stasticmethod for hyperoptimization
    """
    def __init__(self,units,activation,**kwargs):
        super(Perceptron_Layer, self).__init__(**kwargs)
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
            self.R_ListTensor = RListTensor()
            input_shape = self.R_ListTensor.get_output_shape(input_shape)
        self.dense.build(input_shape)

    def call(self,input):
        if isinstance(input,list):
            input = self.R_ListTensor.call(input)
        return self.dense(input)

    def get_config(self):
        config = super(Perceptron_Layer, self).get_config()
        config.update({
            'units': self.units,
            'activation':self.activation
        })
        return config


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
    # Can it be saved ???
    vector_1 = tf.keras.layers.Input(shape=(5, 6, 2))
    vector_3 = tf.keras.layers.Input(shape=(5, 6, 2))
    signallayer = Perceptron_Layer(units=20, activation="MetaActivationLayer")
    vector_2 = [vector_1,vector_3]
    ouputs = signallayer(vector_2)

    model = tf.keras.models.Model(inputs=[vector_1,vector_3], outputs=ouputs)
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
    vector_1 = tf.ones(shape=(12, 5, 6, 2))
    vector_3 = tf.ones(shape=(12, 5, 6, 2))
    print(model_2.predict([vector_1,vector_3]))



