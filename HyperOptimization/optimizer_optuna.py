import optuna
import importlib.util
import tensorflow as tf

from CNN.CustomCNN import CNN_Layer
from MLP.CustomPerceptron import Perceptron_Layer
from RNN.CustomRNN import RNN_Layer
from Transformers.CustomTransformers import CustomTransformerEncoderBlock


import tensorflow as tf

class TransformTensor(tf.keras.layers.Layer):
    def __init__(self):
        super(TransformTensor, self).__init__()

    def call(self, tensor):
        shape = tf.shape(tensor)
        batch_size = shape[0]
        sequence_length = shape[1]
        embedding_size = tf.reduce_prod(shape[2:])
        transformed_tensor = tf.reshape(tensor, (batch_size, sequence_length, embedding_size))

        return transformed_tensor


def OptunaListElements(name_layer,liste,key,trial):
    if type(liste[0])==int:
        if len(liste)==2:
            return trial.suggest_int(f"{name_layer}{key}", liste[0], liste[1], log=True)
        else:
            return trial.suggest_int(f"{name_layer}{key}", liste[0], liste[1], liste[2])
    elif type(liste[0])==str:
        return trial.suggest_categorical(f"{name_layer}{key}", liste)
def loop_initializer(layer,trial,i,j):
    hyperparameters = layer.get_layer_hyperparemeters()
    name_layer = layer.get_name()
    prefix = "hyperparameter"
    liste_hyperparameters = [OptunaListElements(f'{name_layer}_deep_{i}_width_{j}',hyperparameters[key],key.split(".")[-1].replace(prefix,""),trial) for key in hyperparameters.keys()]
    return liste_hyperparameters

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add a channel dimension (for grayscale images)
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
def objective(trial,x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test):
    model = tf.keras.Sequential()


    # Add the convolutional layers and a perceptron
    model.add(CNN_Layer(*loop_initializer(CNN_Layer,trial,1,1)))
    model.add(TransformTensor())
    model.add(RNN_Layer(*loop_initializer(RNN_Layer,trial,1,2)))
    model.add(Perceptron_Layer(*loop_initializer(Perceptron_Layer,trial,1,3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

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

"""
I should create a function which takes in input a layer, then open the dictionnary of hyperparameters and then instances the object and add the hyperparameters in the trial\
I am a bit exhausted to continue but it can be easily done. Use the tips CNN_layer(*list) * unpack elements like lists, tuples, or strings
"""