from utils.utils_optimizer import *
import tensorflow as tf
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


