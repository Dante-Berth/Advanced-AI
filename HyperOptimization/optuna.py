import numpy as np

NETWORK_VERSION = "20.1"

APPLICATION_NAME = f"[HARMONIA OPTIMA v{NETWORK_VERSION}] "

print("Importing other libs...")
# Others libraries
import optuna

import os

import pymysql

pymysql.install_as_MySQLdb()
import MySQLdb

import datetime

# Tensorflow
print("Importing tensorflow...", flush=True)
import tensorflow.keras
from keras.regularizers import l2
import tensorflow as tf
from keras import regularizers

from keras.activations import *

from tensorflow.keras.layers import Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization, Input, Conv1D, \
    AveragePooling1D, Concatenate, LSTM
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
# Normalization
from normalization.harmoniaTwentyNormalization import InputsHarmoniaTwentySequence
import normalization.harmoniaTwentyNormalization as hOneNorm
# AutoML
import AutoML.AutoML
from AutoML.AutoML import loop_RNN_layer
from AutoML.AutoML import loop_MLP_layer
from AutoML.AutoML import CNN_POOLING_layer
from AutoML.AutoML import loop_CNN_layer


def customPrint(to_print, flush=True):
    print(APPLICATION_NAME + " -" + str(os.getpid()) + "- " + str(datetime.datetime.now()) + " " + to_print,
          flush=flush)


max_epochs = 30

nb_epoch_not_improving = 6

customPrint("Creating objective function...")


def objective(trial):
    customPrint("Loading train dataset...", flush=True)
    customPrint("TRIAL NUMBER = " + str(trial.number), flush=True)

    # Paramètres

    # Optuna sur V19 indique une convergence en 72h pour bougie 1h
    h1_hours = 72

    # Optuna sur V19 indique une convergence en 12h pour bougie 5m
    m5_hours = 12 * 12

    # Optuna sur V19 indique une convergence en 1h pour bougie 1m
    m1_hours = 60

    customPrint("Loading test dataset...")

    liste_normalisation = ["MIN_MAX", "LOG", "Z_SCORE", "PARETO_SCALING", "VARIABLE_STABILITY_SCALING", "LOG_Z_SCORE"]
    str_normalisation_price = trial.suggest_categorical('norm_price_method', liste_normalisation)
    str_normalisation_other = trial.suggest_categorical('norm_side_method', liste_normalisation)

    if str_normalisation_price == "MIN_MAX":
        norm_price = hOneNorm.NORMALISATION_MINMAX_WINDOW
    elif str_normalisation_price == "LOG":
        norm_price = hOneNorm.NORMALISATION_LOG
    elif str_normalisation_price == "Z_SCORE":
        norm_price = hOneNorm.NORMALISATION_Z_SCORE
    elif str_normalisation_price == "PARETO_SCALING":
        norm_price = hOneNorm.NORMALISATION_PARETO_SCALING
    elif str_normalisation_price == "VARIABLE_STABILITY_SCALING":
        norm_price = hOneNorm.NORMALISATION_VARIABLE_STABILITY_SCALING
    elif str_normalisation_price == "LOG_Z_SCORE":
        norm_price = hOneNorm.NORMALISATION_LOG_Z_SCORE
    else:
        raise Exception("Erreur: cannot determine norm_price")

    if str_normalisation_other == "MIN_MAX":
        norm_side = hOneNorm.NORMALISATION_MINMAX_WINDOW
    elif str_normalisation_other == "LOG":
        norm_side = hOneNorm.NORMALISATION_LOG
    elif str_normalisation_other == "Z_SCORE":
        norm_side = hOneNorm.NORMALISATION_Z_SCORE
    elif str_normalisation_other == "PARETO_SCALING":
        norm_side = hOneNorm.NORMALISATION_PARETO_SCALING
    elif str_normalisation_other == "VARIABLE_STABILITY_SCALING":
        norm_side = hOneNorm.NORMALISATION_VARIABLE_STABILITY_SCALING
    elif str_normalisation_other == "LOG_Z_SCORE":
        norm_side = hOneNorm.NORMALISATION_LOG_Z_SCORE
    else:
        raise Exception("Erreur: cannot determine norm_price")

    normalization_dependance_price_str = trial.suggest_categorical('normalization_dependance_price',
                                                                   ["independant", "dependant"])

    normalization_independant_price = normalization_dependance_price_str == 'independant'

    test = InputsHarmoniaTwentySequence(hOneNorm.BASE_PATH, hOneNorm.Y_PATH, "ETHUSDT", h1_hours, m5_hours, m1_hours,
                                        norm_price, norm_side, normalization_independant_price, True, 256,
                                        "ETHUSDT-5m_rebuilt_signal_sym7_12_4_Y_area_24_2.5-train.zip",
                                        "5m_rebuilt_signal_sym7_12_4_Y_area_24_2.5",
                                        "2021-11-01 00:00:00",
                                        "2022-04-20 22:00:00")

    batch = trial.suggest_categorical('batch_size', [32, 64, 128])

    start_date = trial.suggest_categorical("start_date", ["2018-04-10 04:00:00", "2020-03-01 04:00:00"])

    # Date
    # attention dans la date à bien mettre une date où il existe des bougies 15m, et à pas mettre une date trop tard pour les Y (len(Y) = len(1h) - 8
    customPrint("Loading train_eth...")
    train = InputsHarmoniaTwentySequence(hOneNorm.BASE_PATH, hOneNorm.Y_PATH, "ETHUSDT", h1_hours, m5_hours, m1_hours,
                                         norm_price, norm_side, normalization_independant_price, True, batch,
                                         "ETHUSDT-5m_rebuilt_signal_sym7_12_4_Y_area_24_2.5-train.zip",
                                         "5m_rebuilt_signal_sym7_12_4_Y_area_24_2.5",
                                         start_date,
                                         "2021-10-29 23:00:00")

    # customPrint("Clearing old session ...")

    # tf.keras.backend.clear_session()

    customPrint("Creating model...", flush=True)
    inputs_1h = Input(shape=(h1_hours, 9))
    h1_layers_CNN = trial.suggest_int('h1_layers_CNN', 1, 5)
    h1_liste_CNN = loop_CNN_layer(inputs_1h, h1_layers_CNN, trial, "CNN_h1")
    h1_liste_RNN_MLP = []
    for i in range(h1_layers_CNN):
        # choisi un mode pour la création du réseau
        network_mode = trial.suggest_int(f'{i}_h1_layers_CNN_boolean', 1, 4)
        # si network_mode est à 4, alors on saute la connexion
        if network_mode == 1:
            h1_layers_RNN = trial.suggest_int(f'{i}_h1_layers_RNN', 1, 5)
            A = loop_RNN_layer(h1_liste_CNN[i], h1_layers_RNN, trial, "RNN_h1")
            h1_liste_RNN_MLP.append(Flatten()(A[-1]))
        if network_mode == 2:
            h1_layers_MLP = trial.suggest_int(f'{i}_h1_layers_MLP', 1, 5)
            B = loop_MLP_layer(h1_liste_CNN[i], h1_layers_MLP, trial, "MLP_h1")
            h1_liste_RNN_MLP.append(Flatten()(B[-1]))
        if network_mode == 3:
            h1_layers_RNN = trial.suggest_int(f'{i}_h1_layers_RNN', 1, 5)
            h1_layers_MLP = trial.suggest_int(f'{i}_h1_layers_MLP', 1, 5)
            A = loop_RNN_layer(h1_liste_CNN[i], h1_layers_RNN, trial, "RNN_h1")
            B = loop_MLP_layer(h1_liste_CNN[i], h1_layers_MLP, trial, "MLP_h1")
            h1_liste_RNN_MLP.append(Flatten()(A[-1]))
            h1_liste_RNN_MLP.append(Flatten()(B[-1]))

    if len(h1_liste_RNN_MLP) > 0:
        Final_concatenation = Concatenate()(h1_liste_RNN_MLP)
    else:
        Final_concatenation = h1_liste_CNN[-1]
    h1_layers_MLP = trial.suggest_int('h1_layers_MLP', 1, 5)
    h1_liste_MLP = loop_MLP_layer(Final_concatenation, h1_layers_MLP, trial, "MLP_h1_last_layer")

    inputs_5m = Input(shape=(m5_hours, 9))
    m5_layers_CNN = trial.suggest_int('m5_layers_CNN', 1, 5)
    m5_liste_CNN = loop_CNN_layer(inputs_5m, m5_layers_CNN, trial, "CNN_m5")
    m5_liste_RNN_MLP = []
    for i in range(m5_layers_CNN):
        boolean = trial.suggest_int(f'{i}_m5_layers_CNN_boolean', 1, 4)
        if boolean == 1:
            m5_layers_RNN = trial.suggest_int(f'{i}_m5_layers_RNN', 1, 5)
            A = loop_RNN_layer(m5_liste_CNN[i], m5_layers_RNN, trial, "RNN_m5")
            m5_liste_RNN_MLP.append(Flatten()(A[-1]))
        if boolean == 2:
            m5_layers_MLP = trial.suggest_int(f'{i}_m5_layers_MLP', 1, 5)
            B = loop_MLP_layer(m5_liste_CNN[i], m5_layers_MLP, trial, "MLP_m5")
            m5_liste_RNN_MLP.append(Flatten()(B[-1]))
        if boolean == 3:
            m5_layers_RNN = trial.suggest_int(f'{i}_m5_layers_RNN', 1, 5)
            m5_layers_MLP = trial.suggest_int(f'{i}_m5_layers_MLP', 1, 5)
            A = loop_RNN_layer(m5_liste_CNN[i], m5_layers_RNN, trial, "RNN_m5")
            B = loop_MLP_layer(m5_liste_CNN[i], m5_layers_MLP, trial, "MLP_m5")
            m5_liste_RNN_MLP.append(Flatten()(A[-1]))
            m5_liste_RNN_MLP.append(Flatten()(B[-1]))

    if len(m5_liste_RNN_MLP) > 0:
        Final_concatenation = Concatenate()(m5_liste_RNN_MLP)
    else:
        Final_concatenation = m5_liste_CNN[-1]
    m5_layers_MLP = trial.suggest_int('m5_layers_MLP', 1, 5)
    m5_liste_MLP = loop_MLP_layer(Final_concatenation, m5_layers_MLP, trial, "MLP_m5_last_layer")

    inputs_1m = Input(shape=(m1_hours, 9))
    m1_layers_CNN = trial.suggest_int('m1_layers_CNN', 1, 5)
    m1_liste_CNN = loop_CNN_layer(inputs_1m, m1_layers_CNN, trial, "CNN_m1")
    m1_liste_RNN_MLP = []
    for i in range(m1_layers_CNN):
        boolean = trial.suggest_int(f'{i}_m1_layers_CNN_boolean', 1, 4)
        if boolean == 1:
            m1_layers_RNN = trial.suggest_int(f'{i}_m1_layers_RNN', 1, 5)
            A = loop_RNN_layer(m1_liste_CNN[i], m1_layers_RNN, trial, "RNN_m1")
            m1_liste_RNN_MLP.append(Flatten()(A[-1]))
        if boolean == 2:
            m1_layers_MLP = trial.suggest_int(f'{i}_m1_layers_MLP', 1, 5)
            B = loop_MLP_layer(m1_liste_CNN[i], m1_layers_MLP, trial, "MLP_m1")
            m1_liste_RNN_MLP.append(Flatten()(B[-1]))
        if boolean == 3:
            m1_layers_RNN = trial.suggest_int(f'{i}_m1_layers_RNN', 1, 5)
            m1_layers_MLP = trial.suggest_int(f'{i}_m1_layers_MLP', 1, 5)
            A = loop_RNN_layer(m1_liste_CNN[i], m1_layers_RNN, trial, "RNN_m1")
            B = loop_MLP_layer(m1_liste_CNN[i], m1_layers_MLP, trial, "MLP_m1")
            m1_liste_RNN_MLP.append(Flatten()(A[-1]))
            m1_liste_RNN_MLP.append(Flatten()(B[-1]))

    if len(m1_liste_RNN_MLP) > 0:
        Final_concatenation = Concatenate()(m1_liste_RNN_MLP)
    else:
        Final_concatenation = m1_liste_CNN[-1]
    m1_layers_MLP = trial.suggest_int('m1_layers_MLP', 1, 5)
    m1_liste_MLP = loop_MLP_layer(Final_concatenation, m1_layers_MLP, trial, "MLP_m1_last_layer")

    concatenate = Concatenate()([Flatten()(m1_liste_MLP[-1]), Flatten()(m5_liste_MLP[-1]), Flatten()(h1_liste_MLP[-1])])

    last_layers_MLP = trial.suggest_int('last_layers_MLP', 1, 5)
    last_liste_MLP = loop_MLP_layer(concatenate, last_layers_MLP, trial, "MLP_last_layer")
    final_dense = Dense(1)(last_liste_MLP[-1])
    ###  Instance du modèle
    model = Model(inputs=[inputs_1h,
                          inputs_5m,
                          inputs_1m], outputs=final_dense)
    # Optimisation
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    str_optim = trial.suggest_categorical("optimizer", ["Adam", "Nadam"])
    optim = None

    if str_optim == "Adam":
        optim = tf.keras.optimizers.Adam
    elif str_optim == "Nadam":
        optim = tf.keras.optimizers.Nadam

    customPrint("Compiling model..")
    model.compile(loss="mse", optimizer=optim(learning_rate=lr), run_eagerly=True)
    customPrint("Training model...")

    best_accr = 10000.0
    not_improving_in_row = 0

    for epoch in range(max_epochs):
        customPrint("New epoch stating on trial " + str(trial.number) + " ...", flush=True)
        model.fit(train, workers=10, use_multiprocessing=True, max_queue_size=200)
        train.reshuffle()
        intermediate_value = model.evaluate(test, workers=10, use_multiprocessing=True, max_queue_size=200)
        intermediate_value = round(intermediate_value, 4)
        customPrint("Valeur intermediaire = " + str(intermediate_value))
        customPrint("Best current = " + str(best_accr), flush=True)
        if np.isnan(intermediate_value):
            # cannot report NaN, then just report None
            try:
                trial.report(None, epoch)
            except:
                break
            raise optuna.TrialPruned()
        if intermediate_value < best_accr:
            customPrint("New model is better, saving it...", flush=True)
            best_accr = intermediate_value
            not_improving_in_row = 0
            model.save(f'best_model_optuna_v{NETWORK_VERSION}-{trial.number}.h5')
            customPrint("Model saved !", flush=True)
        else:
            not_improving_in_row += 1

        customPrint("Reporting to OPTUNA", flush=True)
        trial.report(best_accr, epoch)

        customPrint("Checking pruning...", flush=True)

        if not_improving_in_row >= nb_epoch_not_improving:
            break

        if trial.should_prune():
            raise optuna.TrialPruned()

    customPrint("Done !", flush=True)

    return best_accr


def callback_save_best_model(study, trial):
    try:
        if study.best_trial.number != trial.number:
            customPrint("Current trial is not best trial, removing...")
            os.remove(f'best_model_optuna_v{NETWORK_VERSION}-{trial.number}.h5')
        else:
            customPrint("Current trial is best trial ! Keeping version")
    except Exception as e:
        customPrint("Error while evaluation best trial over trial. Skipping ...")
        customPrint(str(e))
    # we check if we need to end the process of not
    if os.path.exists("stop-{}.optuna".format(os.getpid())):
        customPrint("--- STOPPING STUDY BECAUSE FILE IS PRESENT ---")
        os.remove("stop-{}.optuna".format(os.getpid()))
        study.stop()
    elif os.path.exists("stop-recent.optuna"):
        customPrint("--- STOPPING STUDY BECAUSE FILE IS PRESENT AND FIRST HERE ---")
        os.remove("stop-recent.optuna")
        study.stop()
    elif os.path.exists("stop-all.optuna"):
        customPrint("--- STOPPING STUDY BECAUSE FILE IS PRESENT DOWN ALL ---")
        study.stop()


if __name__ == '__main__':
    study_name = f"study_optuna_v{NETWORK_VERSION}"
    # storage_name = "sqlite:///{}.db".format(study_name)
    storage_name = "mysql://db_adrien_b:0ewOlkRr0ID1CBPv@195.83.10.26/db_adrien_b"
    customPrint("Creating study...")
    study = optuna.create_study(study_name=study_name, pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
                                direction='minimize',
                                storage=storage_name, load_if_exists=True)
    customPrint("Optimizing study...")
    study.optimize(objective, n_trials=200, callbacks=[callback_save_best_model], gc_after_trial=True)
    print("DONE !")
