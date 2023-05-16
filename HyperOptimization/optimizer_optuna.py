import optuna
import importlib.util
import tensorflow as tf

from CNN.CustomCNN import CNN_Layer

name = "alexw" # name user
# Specify the absolute path to the CustomCNN.py file
custom_cnn_path = f'C:/Users/{name}/Documents/Git/AI-WORK/Advanced-Ai/CNN/CustomCNN.py'

# Load the CustomCNN module
spec = importlib.util.spec_from_file_location('CustomCNN', custom_cnn_path)
CustomCNN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CustomCNN)


def OptunaListElements(name_layer,liste,key,trial):
    if type(liste[0])==int:
        if len(liste)==2:
            return trial.suggest_int(f"{name_layer}+{key}", liste[0], liste[1], log=True)
        else:
            return trial.suggest_int(f"{name_layer}+{key}", liste[0], liste[1], liste[2])
    elif type(liste[0])==str:
        return trial.suggest_categorical(f"{name_layer}+{key}", liste)
def loop_initializer(a,trial):
    liste = []
    for key in a.keys():
        liste.append(OptunaListElements("CNN",a[key],key.split(".")[-1],trial))
    return liste

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

    # Add the convolutional layers
    model.add(CNN_Layer(*loop_initializer(CNN_Layer.CNN_layer_hyperparemeters(),trial)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)

    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test)

    # Return the negative accuracy (as Optuna tries to minimize the objective)
    return -accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

"""
I should create a function which takes in input a layer, then open the dictionnary of hyperparameters and then instances the object and add the hyperparameters in the trial\
I am a bit exhausted to continue but it can be easily done. Use the tips CNN_layer(*list) * unpack elements like lists, tuples, or strings
"""