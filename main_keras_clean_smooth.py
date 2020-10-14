#!python3

"""
keras avec des array input_shape=(50, 3)
avec nettoyage des datas et avec ou sans smooth

Les valeurs
    PAQUET = 51
    window = 81 # impair
    polyorder = 3
    smooth = 1
    doivent correpondre à des valeurs de create_smooth_paquets.py
"""

import numpy as np
import random
from datetime import datetime
from tensorflow.keras import Sequential, layers

from create_smooth_paquets import get_one_shot_final_npz


def main():
    """Choisir en commentant une des 2 lignes"""

    # Définir les parametres dans la fonction
    # #hyper_parameter_optimization()

    # Définir les options, le npz doit être créé dans create_smooth_paquets avant
    PAQUET = 49
    window = 81 # impair
    polyorder = 3
    smooth = 1
    epochs = 5

    only_one_train_test(PAQUET=PAQUET, window=window, polyorder=polyorder,
                        smooth=smooth, epochs=epochs)

def only_one_train_test(**kwargs):

    PAQUET = kwargs.get('PAQUET', None)
    window = kwargs.get('window', None)
    polyorder = kwargs.get('polyorder', None)
    smooth = kwargs.get('smooth', None)
    epochs = kwargs.get('epochs', None)

    # Recherche du fichier
    if smooth:
        infile =    './npz_final/hyperparameter/smooth_keras_' +\
                    str(PAQUET) + '_' +\
                    str(window) + '_' +\
                    str(polyorder) +\
                    '.npz'
    else:
        infile =    './npz_final/hyperparameter/no_smooth_keras_' +\
                    str(PAQUET) + '_' +\
                    str(window) + '_' +\
                    str(polyorder) +\
                    '.npz'

    try:
        np.load(infile, allow_pickle=True)
    except:
        # création du npz
        get_one_shot_final_npz(PAQUET=PAQUET, window=window, polyorder=polyorder,
                                smooth=smooth)

    train, test, train_label, test_label = get_train_test_datas(PAQUET=PAQUET,
                                                                window=window,
                                                                polyorder=polyorder,
                                                                smooth=smooth)
    model = build_the_model(PAQUET=PAQUET)
    model = compile_the_model(model)
    model = training_the_model(model, train, train_label, epochs=epochs)
    test_acc = testing_the_model(model, test, test_label)

    a = "Paquets={} window={} polyorder={} Epochs={} smooth={} Efficacité={}%"
    print(a.format(PAQUET, window, polyorder, epochs, smooth, round(test_acc*100, 1)))

def hyper_parameter_optimization():
    """pas de model.save('acc_model.h5')"""

    resp = ""
    smooth = 1

    for i in range(5):
        for PAQUET in [25, 51, 75, 101, 125, 151, 251, 301]:
            for window in [21, 31, 41, 51, 61, 71, 81]:  # toujours impair
                for polyorder in [1, 3, 5, 7, 9]:
                    for epochs in [5, 10, 20]:
                        train, test, train_label, test_label = get_train_test_datas(PAQUET=PAQUET,
                                                                                    window=window,
                                                                                    polyorder=polyorder,
                                                                                    smooth=smooth)
                        model = build_the_model(PAQUET=PAQUET)
                        model = compile_the_model(model)
                        model = training_the_model(model, train, train_label, epochs=epochs)

                        test_acc = testing_the_model(model, test, test_label)

                        a = "Paquets={} window={} polyorder={} Efficacité={}% indice={} Epochs={} smooth={}"
                        print(a.format(PAQUET, window, polyorder, round(test_acc*100, 1), i, epochs, smooth))

                        b = "Paquets={} window={} polyorder={} Efficacité={}% indice={} Epochs={} smooth={}\n"
                        resp += b.format(PAQUET, window, polyorder, round(test_acc*100, 1), i, epochs, smooth)

    print("\n\n\n\n", resp)

    now = datetime.now()
    time_ = now.strftime('%Y_%m_%d_%H_%M')
    outfile = './hyperparameter/hyperparameter_2_hidden_layer' + time_ + '.npz'
    print("Save:", outfile)
    with open(outfile, "w") as fd:
        fd.write(resp)
    fd.close()

def get_train_test_datas(**kwargs):

    PAQUET = kwargs.get('PAQUET', None)
    window = kwargs.get('window', None)
    polyorder = kwargs.get('polyorder', None)
    smooth = kwargs.get('smooth', None)

    print("\n\n\n\nInit ... Chargement des datas smooth_keras_ ....")
    if smooth:
        infile =    './npz_final/hyperparameter/smooth_keras_' +\
                    str(PAQUET) + '_' +\
                    str(window) + '_' +\
                    str(polyorder) +\
                    '.npz'
    else:
        infile =    './npz_final/hyperparameter/no_smooth_keras_' +\
                    str(PAQUET) + '_' +\
                    str(window) + '_' +\
                    str(polyorder) +\
                    '.npz'

    data = np.load(infile, allow_pickle=True)
    train = data["train"]
    test = data["test"]
    train_label = data["train_label"]
    test_label = data["test_label"]

    print("Taille:", train.shape, test.shape, train_label.shape,
                    test_label.shape)

    return train, test, train_label, test_label

def build_the_model(**kwargs):
    print("Build the model ...")

    PAQUET = kwargs.get('PAQUET', None)

    # Choix du model
    model = Sequential()

    # Input layer
    model.add(layers.Dense(units=4, input_shape=(PAQUET, 3)))
    model.add(layers.Flatten())

    # Hiiden layer
    model.add(layers.Dense(8))
    model.add(layers.Dense(64))

    # Output
    model.add(layers.Dense(7))

    print(model.summary())
    print("Build done.")

    return model

def compile_the_model(model, **kwargs):

    print("Compile the model ...")
    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'] )

    print("Compile done.")
    return model

def training_the_model(model, train, train_label, **kwargs):

    epochs = kwargs.get('epochs', None)

    print("Training the model ...")
    model.fit(train, train_label, epochs=epochs)
    print("Training done.")

    return model

def testing_the_model(model, test, test_label, **kwargs):

    print("Testing ......")
    test_loss, test_acc = model.evaluate(test, test_label)
    return test_acc

if __name__ == "__main__":
    main()
