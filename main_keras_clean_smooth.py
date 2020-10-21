#!python3

"""
keras avec des array input_shape=(50, 3)
avec nettoyage des datas et avec ou sans smooth

le npz est créé automatiquement si besoin
git reset --hard origin/master
"""

import os
# Désactivation du GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import random
from datetime import datetime
from tensorflow.keras import Sequential, layers, utils

from create_clean_smooth_paquets import getTrainTestNpz


def main_only_one():

    kwargs = {  "PAQUET": 1200,
                "window": 41, # impair
                "polyorder": 2,
                "save": 1,  # pour faire enreg
                "plot": 0,  # pour afficher les courbes
                "smooth": 1,  # lissage
                "dt": 3000,  # ms d'affichage
                "gliss": 100,  # paquets glissants
                "clean": 0,  # coupe des début fin d'activité
                "fullscreen": 0,
                "epochs": 3
                }

    ka = KerasActivity(**kwargs)
    b = ka.one_training()
    print(b)

def hyper_parameter_optimization():
    """pas de model.save('acc_model.h5')"""

    resp = ""

    kwargs = {  "PAQUET": 900,
                "window": 61, # impair
                "polyorder": 3,
                "save": 1,  # pour faire enreg
                "plot": 0,  # pour afficher les courbes
                "smooth": 1,  # lissage
                "dt": 3000,  # ms d'affichage
                "gliss": 100,  # paquets glissants
                "clean": 1,  # coupe des début fin d'activité
                "fullscreen": 0,
                "epochs": 3
                }

    for paquet in [300, 600, 900]:
        kwargs["PAQUET"] = paquet
        for window in [61, 101]:  # toujours impair
            kwargs["window"] = window
            for polyorder in [3, 6, 9]:
                kwargs["polyorder"] = polyorder
                for epochs in [3]:
                    kwargs["epochs"] = epochs
                    for smooth in [0, 1]:
                        kwargs["smooth"] = smooth
                        for gliss in [20, 50, 100]:
                            kwargs["gliss"] = gliss
                            for clean in [0, 1]:
                                kwargs["clean"] = clean
                                print(kwargs)
                                ka = KerasActivity(**kwargs)
                                a, test_acc = ka.one_training()
                                if test_acc > 0.50:
                                    fichier = (f'./h5/keras_{paquet}_{window}_'
                                    f'{polyorder}_{gliss}_{smooth}_{clean}')
                                    ka.model.save(fichier)
                                resp += a + "\n"
                                print("\n", resp, "\n")
                                del ka

    now = datetime.now()
    time_ = now.strftime('%Y_%m_%d_%H_%M')
    outfile = './hyperparameter/hyperparameter_new_' + time_ + '.txt'
    print("Save:", outfile)
    with open(outfile, "w") as fd:
        fd.write(resp)
    fd.close()

def some_optimization():
    """

    """

    kwargs = {  "PAQUET": 600,
                "window": 61, # impair
                "polyorder": 3,
                "save": 1,  # pour faire enreg
                "plot": 0,  # pour afficher les courbes
                "smooth": 0,  # lissage
                "dt": 3000,  # ms d'affichage
                "gliss": 80,  # paquets glissants
                "clean": 1,  # coupe des début fin d'activité
                "fullscreen": 0,
                "epochs": 3,
                "utils_to_categorical": 1,  # ne marche plus si 0 !
                "loss": 'categorical_crossentropy',  # 'binary_crossentropy'
                "optimizer": 'adam',  # 'SGD',
                "metrics": 'accuracy'}

    resp = ""
    ka = KerasActivity(**kwargs)
    a, test_acc = ka.one_training()
    resp += a + "\n"
    print("\n", resp, "\n")
    del ka

    now = datetime.now()
    time_ = now.strftime('%Y_%m_%d_%H_%M')
    outfile = './hyperparameter/some_optimization_' + time_ + '.txt'
    print("Save:", outfile)
    with open(outfile, "w") as fd:
        fd.write(resp)
    fd.close()


class KerasActivity:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.PAQUET = kwargs.get('PAQUET', None)
        self.window = kwargs.get('window', None)
        self.polyorder = kwargs.get('polyorder', None)
        self.smooth = kwargs.get('smooth', None)
        self.gliss = kwargs.get('gliss', None)
        self.clean = kwargs.get('clean', None)
        self.epochs = kwargs.get('epochs', None)
        self.utils_to_categorical = kwargs.get('utils_to_categorical', None)
        self.loss = kwargs.get('loss', None)
        self.optimizer = kwargs.get('optimizer', None)
        self.metrics = kwargs.get('metrics', None)

        self.model = None

    def one_training(self):

        infile = (f'./npz_final/hyperparameter/keras_{self.PAQUET}_'
                 f'{self.window}_{self.polyorder}_{self.gliss}_'
                 f'{self.smooth}_{self.clean}.npz')
        print(f"\nInit ...")
        print((f'Paquets={self.PAQUET} window={self.window} '
               f'polyorder={self.polyorder} Epochs={self.epochs} '
               f'smooth={self.smooth} gliss={self.gliss} clean={self.clean}'))
        try:
            data = np.load(infile, allow_pickle=True)
        except:
            # création du npz
            gttn = getTrainTestNpz(**self.kwargs)
            data = np.load(infile, allow_pickle=True)

        print(f"Chargement de       {infile}")
        train, test, train_label, test_label = self.get_train_test_datas(data)
        print(f"Vérification des shapes: {train.shape} {test.shape} {train_label.shape} {test_label.shape}")

        self.build_the_model()
        self.compile_the_model()
        self.training_the_model(train, train_label)
        test_acc = self.testing_the_model(test, test_label)

        a = (f'Paquets={self.PAQUET} window={self.window} polyorder={self.polyorder} '
             f'Epochs={self.epochs} smooth={self.smooth} gliss={self.gliss} '
             f'clean={self.clean} Efficacité={round(test_acc*100, 1)} %')
        b = a.replace(".", ",")  # pour Office writter
        return b, test_acc

    def get_train_test_datas(self, data):

        train = data["train"]
        test = data["test"]

        if self.utils_to_categorical:
            train_label = utils.to_categorical(data["train_label"], 7)
            test_label  = utils.to_categorical(data["test_label"], 7)
        else:
            train_label = data["train_label"]
            test_label  = data["test_label"]

        return train, test, train_label, test_label

    def build_the_model(self):
        print("Build the model ...")

        # Choix du model
        self.model = Sequential()

        # Input layer
        self.model.add(layers.Dense(units=4, input_shape=(self.PAQUET, 3)))
        self.model.add(layers.Flatten())

        # Hidden layer
        self.model.add(layers.Dense(8, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))

        # Output
        self.model.add(layers.Dense(7, activation='softmax'))

        print(self.model.summary())

    def compile_the_model(self):
        """ optimizer='sgd' stochastic gradient descent"""

        print("Compile the model ...")
        self.model.compile(loss=self.loss,
                            optimizer=self.optimizer,
                            metrics=['accuracy'])  # self.metrics)

    def training_the_model(self, train, train_label):

        print("Training the model ...")
        self.model.fit(train, train_label, epochs=self.epochs)

    def testing_the_model(self, test, test_label):

        print("Testing ......")
        test_loss, test_acc = self.model.evaluate(test, test_label)

        return test_acc


if __name__ == "__main__":

    # #hyper_parameter_optimization()

    # #main_only_one()

    some_optimization()
