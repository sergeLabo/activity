#!python3


import os
from time import time, sleep
import numpy as np


PAQUET = 50


def get_data_per_npz(num):
    data = np.load('./npz/' + str(num) + '.npz')
    x = data['x']
    y = data['y']
    z = data['z']  # array
    activity = data['activity']

    return x, y, z, activity

def get_only_activity_3(num):
    """Activity = 3, c'est index 2 ???"""

    x, y, z, activity = get_data_per_npz(num)

    acc = []
    for i in range(x.shape[0]):
        if activity[i] == 2:
            a = int((x[i]**2 + y[i]**2 + z[i]**2)**0.5)
            acc.append(a)

    return acc

def get_accs():
    """Toutes les accélérations et activités regroupées"""

    accs = []
    for num in range(1, 16, 1):
        acc = get_only_activity_3(num)
        accs = accs + acc

    accs = np.array(accs)
    print("accs.shape =", accs.shape)
    # #print("Vérif", accs[0])

    return accs

def get_train_test_label(accs):
    """
    accs.shape = (1 900 000, 3)
    50 Hz = 50 valeurs par secondes
    2 secondes = 100 valeurs
    Je fais des paquets de 100 = paquet
    """

    total = accs.shape[0]
    nb_paquet = int(total/PAQUET)
    print(f"Nombre de paquets total possible = {nb_paquet}")
    n_train = int(nb_paquet*0.8)
    n_test = nb_paquet - n_train
    print(f"Nombre de valeurs pour le training = {n_train}")
    print(f"Nombre de valeurs pour le testing = {n_test}")

    train = []
    for i in range(n_train):
        train.append(accs[i*50 : (i+1)*50])
    test = []
    for i in range(n_test):
        test.append(accs[(i+n_train)*50 : (i+1+ n_train)*50])

    train = np.array(train)
    test  = np.array(test)

    train_label = np.array([2]*n_train)
    test_label  = np.array([2]*n_test)

    print("Taille:", train.shape, test.shape, train_label.shape, test_label.shape)
    return train, test, train_label, test_label

def main():

    accs = get_accs()
    train, test, train_label, test_label = get_train_test_label(accs)

    outfile = './datas_only_3.npz'
    np.savez_compressed(outfile, **{"train": train,
                                    "test": test,
                                    "train_label": train_label,
                                    "test_label":  test_label})
    print('Fichier compressé =', outfile)


if __name__ == "__main__":
    main()
