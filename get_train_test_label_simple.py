#!/usr/bin/env python3


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

def get_accs_acts():
    """Toutes les accélérations et activités regroupées"""
    for num in range(1, 16, 1):
        x, y, z, activity = get_data_per_npz(num)  # tout est array
        x_y_z = np.vstack((x, y,z))
        print("x_y_z.shape =", x_y_z.shape)
        if num == 1:
            accs = x_y_z
            acts = activity
        else:
            accs = np.concatenate((accs, x_y_z), axis=1)
            acts = np.concatenate((acts, activity))
    accs = np.transpose(accs)
    acts = np.transpose(acts)
    print("accs.shape =", accs.shape)
    print("acts.shape =", acts.shape)
    print("Vérif", accs[0], acts[0])

    return accs, acts

def get_train_test_label(accs, acts):
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

    train_test = []
    train_test_label = []
    paquet = []
    # Activité en cours= 1
    en_cours = 0
    # Parcours des 1900000 lignes de 4
    for i in range(total):
        if len(paquet) < PAQUET:
            # Si l'activité reste la même, je fais un paquet de 100
            if acts[i] == en_cours:
                paquet.append(int((accs[i][0]**2 + accs[i][1]**2 +accs[i][2]**2)**0.5))
            # Changement d'activité
            else:
                paquet = [int((accs[i][0]**2 + accs[i][1]**2 +accs[i][2]**2)**0.5)]
                en_cours = acts[i]

        # Ajout du paquet
        else:
            if len(paquet) != PAQUET:
                print("error")
            train_test.append(paquet)
            train_test_label.append(en_cours)
            paquet = []

    train = np.array(train_test)[:n_train, ]
    test  = np.array(train_test)[n_train:, ]
    train_label = np.array(train_test_label)[:n_train, ]
    test_label = np.array(train_test_label)[n_train:, ]
    print("Taille", train.shape, test.shape, train_label.shape, test_label.shape)
    return train, test, train_label, test_label

def main():

    accs, acts = get_accs_acts()
    train, test, train_label, test_label = get_train_test_label(accs, acts)

    outfile = './datas_numpy.npz'
    np.savez_compressed(outfile, **{"train": train,
                                    "test": test,
                                    "train_label": train_label,
                                    "test_label":  test_label})
    print('Fichier compressé =', outfile)


if __name__ == "__main__":
    main()


    # #train = np.array(train)
    # #test = np.array(test)
    # #train_label = np.array(train_label)
    # #test_label = np.array(test_label)

    # #train = accs[:n_train, ]
    # #test = accs[n_train:, ]
    # #train_label = acts[:n_train, ]
    # #test_label = acts[n_train:, ]
