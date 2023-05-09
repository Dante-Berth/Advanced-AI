import optuna
import importlib.util
import tensorflow as tf
name = "alexw" # name user
# Specify the absolute path to the CustomCNN.py file
custom_cnn_path = f'C:/Users/{name}/Documents/Git/AI-WORK/Advanced-Ai/CNN/CustomCNN.py'

# Load the CustomCNN module
spec = importlib.util.spec_from_file_location('CustomCNN', custom_cnn_path)
CustomCNN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CustomCNN)


a = CustomCNN.CNN_Layer.CNN_layer_hyperparemeters()
print(a)

def Optuna_Listelements(name_layer,liste,key,trial):
    if type(liste[0])==int:
        if len(liste)==2:
            return trial.suggest_loguniform(f"{name_layer}+{key}", liste[0], liste[1])
        else:
            return trial.suggest_int(f"{name_layer}+{key}", liste[0], liste[1], liste[2])
    elif type(liste[0])==str:
        return trial.suggest_categorical(f"{name_layer}+{key}", liste)
def
    return CustomCNN.CNN_Layer()