import random
import sys

import optuna
import os

from CNN.CustomCNN import CNN_Layer
from MLP.CustomPerceptron import Perceptron_Layer
from RNN.CustomRNN import RNN_Layer
#from Transformers.CustomTransformers import TransformerEncoderBlock_layer, R_TransformerEncoderBlock_layer
from Layers.CustomLayers import SignalLayer, LinalgMonolayer
from Fromtwotensorsintoonetensor import RListTensor
from Transformers.CustomMultiHeadAttention import MultiHeadAttention_Layer
from Layers.ReductionLayers import ReductionLayerSVD,ReductionLayerPooling
import tensorflow as tf
# Load the MNIST dataset
# Adding a new layer called MultiHeadAttention
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
def loop_initializer(layer,trial,i,j,timestamp=None):
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
    liste_hyperparameters = [OptunaListElements(f'{timestamp}_{name_layer}_deep_{i}_width_{j}',hyperparameters[key],key.split(".")[-1].replace(prefix,""),trial) for key in hyperparameters.keys()]
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
    #@staticmethod
    #def Transformer(trial,i,j,x,y=None):
    #    if y is not None:
    #        return R_TransformerEncoderBlock_layer(*loop_initializer(R_TransformerEncoderBlock_layer, trial, i, j))([x,y])
    #    else:
    #        return R_TransformerEncoderBlock_layer(*loop_initializer(R_TransformerEncoderBlock_layer, trial, i, j))([x,x])
    @staticmethod
    def CNN(trial,i,j,x,y=None,timestamp=None):
        if y is not None:
            return CNN_Layer(*loop_initializer(CNN_Layer, trial, i, j, timestamp))([x,y])
        else:
            return CNN_Layer(*loop_initializer(CNN_Layer, trial, i, j,timestamp))(x)
    @staticmethod
    def MLP(trial,i,j,x,y=None,timestamp=None):
        if y is not None:
            return Perceptron_Layer(*loop_initializer(Perceptron_Layer, trial, i, j, timestamp))([x,y])
        else:
            return Perceptron_Layer(*loop_initializer(Perceptron_Layer, trial, i, j, timestamp))(x)
    @staticmethod
    def RNN(trial,i,j,x,y=None,timestamp=None):
        if y is not None:
            return RNN_Layer(*loop_initializer(RNN_Layer, trial, i, j, timestamp))([x,y])
        else:
            return RNN_Layer(*loop_initializer(RNN_Layer, trial, i, j, timestamp))(x)

    @staticmethod
    def MHA(trial, i, j, x, y=None,timestamp=None):
        if y is not None:
            return MultiHeadAttention_Layer(*loop_initializer(MultiHeadAttention_Layer, trial, i, j, timestamp))([x, y])
        else:
            return MultiHeadAttention_Layer(*loop_initializer(MultiHeadAttention_Layer, trial, i, j, timestamp))(x)

    @staticmethod
    def ReductionLayerSVD(trial, i, j, x, y=None,timestamp=None):
        if y is not None:
            return ReductionLayerSVD(*loop_initializer(ReductionLayerSVD, trial, i, j, timestamp))([x, y])
        else:
            return ReductionLayerSVD(*loop_initializer(ReductionLayerSVD, trial, i, j, timestamp))(x)

    @staticmethod
    def ReductionLayerPooling(trial, i, j, x, y=None,timestamp=None):
        if y is not None:
            return ReductionLayerPooling(*loop_initializer(ReductionLayerPooling, trial, i, j, timestamp))([x, y])
        else:
            return ReductionLayerPooling(*loop_initializer(ReductionLayerPooling, trial, i, j, timestamp))(x)

    @staticmethod
    def Fourrier(trial, i, j, x, y=None,timestamp=None):
        if y is not None:
            return SignalLayer(*loop_initializer(SignalLayer, trial, i, j, timestamp))([x,y])
        else:
            return SignalLayer(*loop_initializer(SignalLayer, trial, i, j, timestamp))(x)

    @staticmethod
    def Linalg(trial, i, j, x, y=None,timestamp=None):
        if y is not None:
            return LinalgMonolayer(*loop_initializer(LinalgMonolayer, trial, i, j, timestamp))([x, y])
        else:
            return LinalgMonolayer(*loop_initializer(LinalgMonolayer, trial, i, j, timestamp))(x)

    @staticmethod
    def Residual(trial, i, j, x, y=None,timestamp=None):
        if y is not None:
            return RListTensor()([x, y])
        else:
            return x


    @staticmethod
    def weighted_layer(trial, i, j, x, y=None, only_layers=None,timestamp=None):
        if only_layers is None or only_layers == [None]:
            only_layers = ["RNN", "MLP", "CNN", "MHA"]
        if len(only_layers) == 1:
            name_layer = only_layers[0]
        else:
            name_layer = trial.suggest_categorical(f"{timestamp}_weighted_layer_{i}_{j}", only_layers)
        z = getattr(final_layer,name_layer)(trial,i,j,x,y, timestamp)
        return z

    @staticmethod
    def unweighted_layer(trial, i, j, x, y=None, only_layers=None,timestamp=None):
        if only_layers is None or only_layers == [None]:
            only_layers = ["Fourrier", "Linalg"]
        if len(only_layers)==1:
            name_layer = only_layers[0]
        else:
            name_layer = trial.suggest_categorical(f"unweighted_layer_{i}_{j}", only_layers)
        z = getattr(final_layer, name_layer)(trial, i, j, x, y, timestamp)
        return z
    @staticmethod
    def reduction_layer(trial, i, j, x, y=None, only_reduction_layers=None,timestamp=None):
        if only_reduction_layers is None:
            only_reduction_layers = ["ReductionLayerPooling"]

        if len(only_reduction_layers)==1:
            name_layer = only_reduction_layers[0]
        else:
            name_layer = trial.suggest_categorical(f"reduction_layer_{i}_{j}", only_reduction_layers)
        z = getattr(final_layer, name_layer)(trial, i, j, x, y, timestamp)
        return z

    def loop_weighted_layer(self,trial,i,j,input_tensor,timestamp,possible_layers=None):
        if possible_layers is None:
            same_layer = trial.suggest_categorical(f"same_layer",["True","False"])
            if same_layer == "True":
                    possible_layers = ["RNN", "MLP", "CNN", "MHA"]
                    only_layers = trial.suggest_categorical(f'{timestamp}_{i}_{j}_loop_layer', possible_layers)
            else:
                only_layers = None
        else:
            only_layers = trial.suggest_categorical(f'{timestamp}_{i}_{j}_loop_layer', possible_layers)

        nb_layers =  trial.suggest_int(f'{timestamp}_{i}_{j}_loop_{only_layers}_nb_layer', 1, 8)
        list_outputs = []
        list_outputs.append(input_tensor)
        for num in range(nb_layers):
            x = self.weighted_layer(trial,i,j+num+1,list_outputs[-1],y=None,only_layers=[only_layers], timestamp=timestamp)
            list_outputs.append(x)
        return list_outputs,i,j+nb_layers

    def loop_unweighted_layer(self, trial, i, j, input_tensor, timestamp,possible_layers=None):
        if possible_layers is None:
            same_layer = trial.suggest_categorical(f"same_layer", ["True", "False"])
            if same_layer == "True":
                possible_layers =  ["Fourrier", "Linalg"]
                only_layers = trial.suggest_categorical(f'{timestamp}_{i}_{j}_loop_layer', possible_layers)
            else:
                only_layers = None
        else:
            only_layers = trial.suggest_categorical(f'{timestamp}_{i}_{j}_loop_layer', possible_layers)

        nb_layers = trial.suggest_int(f'{timestamp}_{i}_{j}_loop_{only_layers}_nb_layer', 1, 5)
        list_outputs = []
        list_outputs.append(input_tensor)
        for num in range(nb_layers):
            x = self.unweighted_layer(trial, i, j+num+1, list_outputs[-1], y=None, only_layers=[only_layers], timestamp=timestamp)
            list_outputs.append(x)
        return list_outputs,i,j+nb_layers



class encoder_temporal_layer:
    @staticmethod
    def temporal_layer_data(trial, i, j, input_tensor, list_tensors=None):
        list_outputs = []
        stochastic = trial.suggest_int(f"stochastic_nas_layer_{i}_{j}", 0, 1, step=1)
        reduction_layer = trial.suggest_int(f"reduction_layer_{i}_{j}", 0, 1, step=1)
        if isinstance(list_tensors, tf.Tensor):
            list_y = [list_tensors]
        # CNN or RNN tries to encode the information
        num_layers = trial.suggest_int(f"num_layers_{i}_{j}", 1, 10)

        tensor_x = final_layer.weighted_layer(trial, i - 0.75, j, x=input_tensor,
                                       only_layers=["MHA"])
        tensor_x = final_layer.weighted_layer(trial, i - 0.5, j, x=tensor_x,
                                              only_layers=["RNN"])
        tensor_x = final_layer.weighted_layer(trial, i - 0.25, j, x=tensor_x,
                                              only_layers=["RNN"])
        tensor_z = final_layer.weighted_layer(trial, i - 0.15, j, x=tensor_x,
                                              only_layers=None)
        list_outputs.append(tensor_z)

        tensor_x = input_tensor
        for num_layer in range(num_layers):
            if list_tensors is None:
                tensor_y = None
            else:
                if stochastic:
                    index_list_tensors = random.randint(0, len(list_tensors) - 1)
                else:
                    index_list_tensors = -1
                tensor_y = list_tensors[index_list_tensors]
            tensor_x = final_layer.Residual(trial, i, j, tensor_x, y=final_layer.unweighted_layer(trial, i + num_layer, j, x=tensor_x, only_layers=None))
            tensor_x = final_layer.weighted_layer(trial, i + num_layer, j, x=tensor_x, y=tensor_y, only_layers=["CNN","MLP"])
            if reduction_layer:
                tensor_x = final_layer.reduction_layer(trial, i + num_layer + 0.5, j, x=tensor_x,
                                                only_reduction_layers=["ReductionLayerPooling"])

        tensor_x = final_layer.weighted_layer(trial, i + num_layer + 1, j,x=tensor_x,
                                       only_layers=["MHA"])
        list_outputs.append(tensor_x)


        x = final_layer.weighted_layer(trial, i +num_layer + 2, j, tensor_x, tensor_z,
                                       only_layers=["MLP"])
        x = final_layer.weighted_layer(trial, i + num_layer + 3, j, x, None,
                                       only_layers=["MLP"])
        x = final_layer.weighted_layer(trial, i + num_layer + 4, j, x, None,
                                       only_layers=["MLP"])
        list_outputs.append(x)


        return list_outputs

    @staticmethod
    def loop_layer(trial, i, j, input_tensor,only_layers =None):
        list_outputs = [input_tensor]
        num_layers = trial.suggest_int(f"num_layers_{i}_{j}", 1, 10)
        for num_layer in range(num_layers):
            list_outputs.append(final_layer.weighted_layer(trial, i + num_layer, j, x=list_outputs[-1], y=None, only_layers=only_layers))
        return list_outputs
class usefull_master_layer:
    @staticmethod
    def layer_loop(trial, i, j, x, list_y=None):
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
        print("stochastic",stochastic)
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
    def master_unweighted_layer_list_tensors(trial,i,j,list_tensors,only_layers=None):
        list_outputs = []
        for k in range(len(list_tensors)):
            tensor = list_tensors[k]
            list_outputs.append(tensor)
            bool = trial.suggest_int(f"bool_unweighted_layer_{i}_{j+k}",0,1,step=1)
            if bool:
                list_outputs.append(final_layer.unweighted_layer(trial,i,j+k,tensor,only_layers))
        del list_tensors
        return list_outputs

def nas_simple_nn(input,trial,timestamp:str):
    final_layer_instance = final_layer()
    liste_layer,reseau_a,deep_a = final_layer_instance.loop_weighted_layer(trial=trial,i=0,j=0,input_tensor=input,timestamp=timestamp,possible_layers=None)
    liste_encoder_layer = []
    for a in range(len(liste_layer)):
        loop_chosen = trial.suggest_categorical(f"{timestamp}_boolean",["loop_weighted_layer","loop_unweighted_layer"])
        liste_layer_i = getattr(final_layer_instance,loop_chosen)(trial,reseau_a+a+1,a,input_tensor=liste_layer[a],timestamp=timestamp)[0]
        liste_encoder_layer.append(tf.keras.layers.Flatten()(liste_layer_i[-1]))
    concatenation_encoder_cnn = tf.keras.layers.Concatenate()(liste_encoder_layer)
    final_output,reseau_a,deep_a = final_layer_instance.loop_weighted_layer(trial=trial,i=reseau_a+len(liste_layer)+2,j=0,input_tensor=concatenation_encoder_cnn,timestamp=timestamp,possible_layers=["MLP"])
    return final_output

class multiple_simple_nn_3d:
    def __init__(self,dictionnary:dict,trial):
        super(multiple_simple_nn_3d).__init__()
        layer_instance = final_layer()
        list_inputs = []
        list_outputs = []
        for key in dictionnary:
            input = tf.keras.Input(shape=dictionnary[key]["input_size"])
            list_inputs.append(input)
            out = nas_simple_nn(input,trial,dictionnary[key]["timestamp"])
            list_outputs.append(out)

        list_tensors = [item for sublist in list_outputs for item in sublist]
        concatenation = tf.keras.layers.Concatenate(axis=1)(list_tensors)
        mlp = layer_instance.loop_weighted_layer(trial=trial, i=0, j=0,
                                        input_tensor=concatenation, timestamp="final_timestamp",
                                        possible_layers=["MLP"])[0]
        output = tf.keras.layers.Dense(10,activation="softmax")(mlp[-1])

        self.model = tf.keras.Model(inputs=list_inputs, outputs=output)
    def return_model(self):
        return self.model



def objective(trial,x_train=x_train[:30],y_train=y_train[:30],x_test=x_test[:30],y_test=y_test[:30],width=10, depth=10):

    dictionnary = {"X_1":{"input_size":(28,28,1),"timestamp":"1m"},
                   "X_2": {"input_size": (28, 28, 1), "timestamp": "5m"},
                   "X_3": {"input_size": (28, 28, 1), "timestamp": "1h"},
                   }
    meta_archi = multiple_simple_nn_3d(dictionnary,trial)
    model = meta_archi.return_model()
    model.summary()

    #Optimizer part
    hyperparameters = {
        "learning_rate" : ["0.1", "0.5", "1", "5", "10", "50", "100", "200", "500", "1000"],
        "use_ema" : ["1","5","10","50"],
        "decay_steps": ["10", "50", "100", "200", "500", "1000", "10000"]

    }
    learning_rate = float(trial.suggest_categorical("learning_rate",hyperparameters["learning_rate"]))*1e-4
    use_ema = trial.suggest_categorical("use_ema",hyperparameters["use_ema"])
    ema_momentum = trial.suggest_float('ema_momentum', 0.9, 0.99)
    ema_overwrite_frequency = trial.suggest_int('ema_overwrite_frequency', 1, 100,5)*10
    decay_steps = int(trial.suggest_categorical("decay_steps",hyperparameters["decay_steps"]))

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = learning_rate, decay_steps = decay_steps)

    ranger = tf.keras.optimizers.Adam(learning_rate=lr_schedule, use_ema=use_ema,ema_momentum=ema_momentum, ema_overwrite_frequency=ema_overwrite_frequency)

    model.compile(
        optimizer=ranger,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.summary()

    # Save the Keras model with the unique filename
    model.save(f'best_model_optuna_v12_{trial.number}.h5')
    model = tf.keras.models.load_model(f'best_model_optuna_v12_{trial.number}.h5')

    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=1)

    _, accuracy = model.evaluate(x_test, y_test)

    return accuracy
study = optuna.create_study(direction="maximize",pruner=optuna.pruners.PercentilePruner(percentile=0.3,n_warmup_steps=12))
study.optimize(objective, n_trials=100)

