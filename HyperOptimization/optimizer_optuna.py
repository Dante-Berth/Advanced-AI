import optuna
import importlib.util
import tensorflow as tf
from CNN.CustomCNN import CNN_Layer
from MLP.CustomPerceptron import Perceptron_Layer
from RNN.CustomRNN import RNN_Layer, R_RNN_Layer, Reshape_Layer_3D
from Transformers.CustomTransformers import TransformerEncoderBlock_layer, R_TransformerEncoderBlock_layer

import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension (for grayscale images)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
def OptunaListElements(name_layer,liste,key,trial):
    """
    Generates an optimal value for a hyperparameter using Optuna based on the given list.

    Args:
        name_layer (str): The name of the layer.
        liste (list): The list of values for the hyperparameter.
        key (str): The key representing the hyperparameter.
        trial (optuna.Trial): The Optuna trial object used for optimization.

    Returns:
        The suggested optimal value for the hyperparameter based on Optuna.

    Raises:
        None
    """
    if type(liste[0])==int:
        if len(liste)==2:
            return trial.suggest_int(f"{name_layer}{key}", liste[0], liste[1], log=True)
        else:
            return trial.suggest_int(f"{name_layer}{key}", liste[0], liste[1], liste[2])
    elif type(liste[0])==str:
        return trial.suggest_categorical(f"{name_layer}{key}", liste)
def loop_initializer(layer,trial,i,j):
    """
    Initializes hyperparameters for a layer by suggesting optimal values using Optuna.

    Args:
        layer (tf.keras.layers.Layer): The layer for which hyperparameters are being initialized.
        trial (optuna.Trial): The Optuna trial object used for optimization.
        i (int): Index of the deep loop.
        j (int): Index of the width loop.

    Returns:
        List of suggested optimal values for the layer's hyperparameters.

    Raises:
        None
    """
    hyperparameters = layer.get_layer_hyperparemeters()
    name_layer = layer.get_name()
    prefix = "hyperparameter"
    liste_hyperparameters = [OptunaListElements(f'{name_layer}_deep_{i}_width_{j}',hyperparameters[key],key.split(".")[-1].replace(prefix,""),trial) for key in hyperparameters.keys()]
    return liste_hyperparameters

class Reshape_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Reshape_Layer, self).__init__()
        self.reshape = None

    def build(self, input_shape):
        """
        Builds the Reshape layer.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            None

        Raises:
            None
        """
        if len(input_shape)>3:
            self.reshape = tf.keras.layers.Reshape((input_shape[1], tf.reduce_prod(input_shape[2:]).numpy()[()]))
    def call(self, inputs, *args, **kwargs):
        """
        Performs the forward pass of the Reshape layer.

        Args:
            inputs: The input tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The reshaped tensor if reshape is not None, else returns the input tensor.

        Raises:
            None
        """
        if self.reshape:
            return self.reshape(inputs)
        else:
            return inputs
def objective(trial,x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test):
    input_layer = tf.keras.layers.Input(shape=(28, 28, 1))

    x = CNN_Layer(*loop_initializer(CNN_Layer, trial, 1, 1))(input_layer)

    y = CNN_Layer(*loop_initializer(CNN_Layer, trial, 1, 1))(x)


    x = R_RNN_Layer(*loop_initializer(R_RNN_Layer, trial, 1, 2))(x)

    y = R_TransformerEncoderBlock_layer(*loop_initializer(R_TransformerEncoderBlock_layer, trial, 1, 3))([x,y])

    y = R_RNN_Layer(*loop_initializer(R_RNN_Layer,trial,1,4))([y,x])

    y = Perceptron_Layer(*loop_initializer(Perceptron_Layer, trial, 1, 5))([y,x])

    # Flatten the output
    y = tf.keras.layers.Flatten()(x)

    # Define the output layer
    output = tf.keras.layers.Dense(10, activation="softmax")(y)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)

    _, accuracy = model.evaluate(x_test, y_test)

    return -accuracy
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

