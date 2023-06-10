import random
import tensorflow_models as tfm
import optuna
import importlib.util
import tensorflow as tf
from CNN.CustomCNN import CNN_Layer
from MLP.CustomPerceptron import Perceptron_Layer
from RNN.CustomRNN import RNN_Layer, R_RNN_Layer, Reshape_Layer_3D
from Transformers.CustomTransformers import TransformerEncoderBlock_layer, R_TransformerEncoderBlock_layer
from Layers.CustomLayers import SignalLayer, LinalgMonolayer
from Fromtwotensorsintoonetensor import R_ListTensor
from Layers.ReductionLayers import ReductionLayerSVD,ReductionLayerPooling
import tensorflow as tf
import tensorflow_addons as tfa
from Optimizers.CustomOptimizer import AdaBelief_optimizer
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

class final_layer:
    """
    final_layer is the class containing the functions for building the Meta AI, among these functions they are\
    there are two components, the first one is dedicated to the weighted layers while the second one referred to \
    to unweighted layers which are important for any unusual Deep Learning transformations.
    """
    @staticmethod
    def Transformer(trial,i,j,x,y=None):
        if y is not None:
            return R_TransformerEncoderBlock_layer(*loop_initializer(R_TransformerEncoderBlock_layer, trial, i, j))([x,y])
        else:
            return R_TransformerEncoderBlock_layer(*loop_initializer(R_TransformerEncoderBlock_layer, trial, i, j))([x,x])
    @staticmethod
    def CNN(trial,i,j,x,y=None):
        if y is not None:
            return CNN_Layer(*loop_initializer(CNN_Layer, trial, i, j))([x,y])
        else:
            return CNN_Layer(*loop_initializer(CNN_Layer, trial, i, j))(x)
    @staticmethod
    def MLP(trial,i,j,x,y=None):
        if y is not None:
            return Perceptron_Layer(*loop_initializer(Perceptron_Layer, trial, i, j))([x,y])
        else:
            return Perceptron_Layer(*loop_initializer(Perceptron_Layer, trial, i, j))(x)
    @staticmethod
    def RNN(trial,i,j,x,y=None):
        if y is not None:
            return R_RNN_Layer(*loop_initializer(R_RNN_Layer, trial, i, j))([x,y])
        else:
            return R_RNN_Layer(*loop_initializer(R_RNN_Layer, trial, i, j))(x)


    @staticmethod
    def ReductionLayerSVD(trial, i, j, x, y=None):
        if y is not None:
            return ReductionLayerSVD(*loop_initializer(ReductionLayerSVD, trial, i, j))([x, y])
        else:
            return ReductionLayerSVD(*loop_initializer(ReductionLayerSVD, trial, i, j))(x)

    @staticmethod
    def ReductionLayerPooling(trial, i, j, x, y=None):
        if y is not None:
            return ReductionLayerPooling(*loop_initializer(ReductionLayerPooling, trial, i, j))([x, y])
        else:
            return ReductionLayerPooling(*loop_initializer(ReductionLayerPooling, trial, i, j))(x)

    @staticmethod
    def Fourrier(trial, i, j, x, y=None):
        if y is not None:
            return SignalLayer(*loop_initializer(SignalLayer, trial, i, j))([x,y])
        else:
            return SignalLayer(*loop_initializer(SignalLayer, trial, i, j))(x)

    @staticmethod
    def Linalg(trial, i, j, x, y=None):
        if y is not None:
            return LinalgMonolayer(*loop_initializer(LinalgMonolayer, trial, i, j))([x, y])
        else:
            return LinalgMonolayer(*loop_initializer(LinalgMonolayer, trial, i, j))(x)

    @staticmethod
    def Residual(trial, i, j, x, y=None):
        if y is not None:
            return R_ListTensor()([x,y])
        else:
            return x


    @staticmethod
    def weighted_layer(trial, i, j, x, y=None, only_layers=None):
        if only_layers is None:
            only_layers = ["Transformer", "RNN", "MLP", "CNN"]
        if len(only_layers) == 1:
            name_layer = only_layers[0]
        else:
            name_layer = trial.suggest_categorical(f"weighted_layer_{i}_{j}", only_layers)
        z = getattr(final_layer,name_layer)(trial,i,j,x,y)
        return z

    @staticmethod
    def unweighted_layer(trial, i, j, x, y=None, only_layers=None):
        if only_layers is None:
            only_layers = ["Fourrier", "Linalg"]
        if len(only_layers)==1:
            name_layer = only_layers[0]
        else:
            name_layer = trial.suggest_categorical(f"unweighted_layer_{i}_{j}", only_layers)
        z = getattr(final_layer, name_layer)(trial, i, j, x, y)
        return z
    @staticmethod
    def reduction_layer(trial, i, j, x, y=None, only_reduction_layers=None):
        if only_reduction_layers is None:
            only_reduction_layers = ["ReductionLayerSVD", "ReductionLayerPooling"]

        if len(only_reduction_layers)==1:
            name_layer = only_reduction_layers[0]
        else:
            name_layer = trial.suggest_categorical(f"reduction_layer_{i}_{j}", only_reduction_layers)
        z = getattr(final_layer, name_layer)(trial, i, j, x, y)
        return z





class loop_final_layer:
    @staticmethod
    def layer_loop(trial, i, j, x, list_y=None, only_layers=None):
        """

        :param trial: trial object
        :param i: int
        :param j: int
        :param x: tf.Tensor
        :param list_y: list of tf.Tensor
        :return: list of tf.Tensor
        """
        list_outputs = []
        stochastic = trial.suggest_int(f"stochastic_nas_layer_{i}_{j}",0,1,step=1)
        if list_y is None:
            list_y = [None]
        if isinstance(list_y, tf.Tensor):
            list_y = [list_y]
        skipped_layer = trial.suggest_int(f"skipped_layer_{i}_{j}",0,1,step=1)
        if skipped_layer:
            list_outputs.append(x)
            return list_outputs
        else:
            num_layers = trial.suggest_int(f"num_layers_{i}_{j}", 1, 5)

            for num_layer in range(num_layers):
                if stochastic:
                    index_list_y = random.randint(0,len(list_y)-1)
                else:
                    index_list_y = -1

                x = final_layer.weighted_layer(trial, i+num_layer, j, x, list_y[index_list_y],only_layers)
                x = final_layer.reduction_layer(trial, i+num_layer, j, x, list_y[index_list_y],only_reduction_layers=["ReductionLayerPooling"])

                list_outputs.append(x)


            return list_outputs

    @staticmethod
    def unweighted_layer_list_tensors(trial,i,j,list_tensors,only_layers=None):
        list_outputs = []
        for k in range(len(list_tensors)):
            tensor = list_tensors[k]
            list_outputs.append(tensor)
            bool = trial.suggest_int(f"bool_unweighted_layer_{i}_{j+k}",0,1,step=1)
            if bool:
                list_outputs.append(final_layer.unweighted_layer(trial,i,j+k,tensor,only_layers))
        del list_tensors
        return list_outputs





def objective(trial,x_train=x_train[:30],y_train=y_train[:30],x_test=x_test[:30],y_test=y_test[:30],width=10, depth=10):

    input_layer = tf.keras.layers.Input(shape=(28, 28, 1))

    dictionnary = {}
    width = 1
    depth = 1
    x = loop_final_layer.unweighted_layer_list_tensors(trial,1,1,[input_layer])
    x = loop_final_layer.layer_loop(trial,1,1,x[-1],x,only_layers=["CNN","Transformer","MLP"])
    x = loop_final_layer.layer_loop(trial, 1, 1, x[-1],x,only_layers=["CNN"])




    # Flatten the output
    x = tf.keras.layers.Flatten()(x[-1])


    output = tf.keras.layers.Dense(10, activation="softmax")(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    #ranger = AdaBelief_optimizer.init(*(loop_initializer(AdaBelief_optimizer, trial, -1, -1) + [32, 1000]))
    learning_rate = int(trial.suggest_categorical("learning_rate",["0.1","0.5","1","5","10","50","100","200","500","1000"]))*1e-4
    use_ema = trial.suggest_categorical("use_ema",[True,False])
    clip_norm = int(trial.suggest_categorical("clip_norm",["1","5","10","50"]))*0.1
    clip_value = int(trial.suggest_categorical("clip_value",["1","5","10","50"]))*0.1
    ema_momentum = trial.suggest_float('ema_momentum', 0.9, 0.99)
    ema_overwrite_frequency = trial.suggest_int('ema_overwrite_frequency', 1, 100,5)*10

    ranger = tf.keras.optimizers.Lion(learning_rate=learning_rate, use_ema=use_ema, clip_norm=clip_norm, clip_value=clip_value,ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency)

    model.compile(
        optimizer=ranger,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=1)

    _, accuracy = model.evaluate(x_test, y_test)

    return accuracy
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

