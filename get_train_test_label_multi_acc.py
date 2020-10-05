#!/usr/bin/env python3


import os
from time import time, sleep
import numpy as np

PAQUET = 50

Correction = [  (76.32921230769239, 1.7136738461540517, -71.21460307692314),
                (-94.90996217264092, 55.78574177200471, 52.262010731062674),
                (54.4825972249364, 15.895876490130831, 58.41950361539966),
                (-66.15108837970547, 4.684836333879048, -54.087070376432166),
                (72.72101249999992, 31.33008125000015, -7.769037499999968),
                (-61.926188952868415, 85.71731001635044, 65.12092130518226),
                (-92.1657361963189, 22.818196319018625, 104.91190797546005),
                (182.91347291266015, 110.67535832214526, 47.71776914982411),
                (-97.75267497251753, -157.39261634298282, -70.54224990839134),
                (-83.38591482649827, -123.41551261829636, 19.3696687697161),
                (-5.4582862613690395, 32.302900909526215, -51.09786500718042),
                (122.27138385890271, 45.194645207975555, -24.973792730664854),
                (-14.766814486326666, 4.249150036954688, 3.257147080561708),
                (117.15042204995689, 10.024573643410804, -52.26593453919031),
                (-99.78375845410619, -148.46936231884047, -42.35442512077293)]


def get_data_per_npz(num):
    data = np.load('./npz/' + str(num) + '.npz')
    x = data['x']
    y = data['y']
    z = data['z']  # array
    activity = data['activity']

    xc = x + int(Correction[num-1][0])
    yc = y + int(Correction[num-1][1])
    zc = z + int(Correction[num-1][2])

    return xc, yc, zc, activity


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

    [
    [42, 56, 12],
    [58, 23, 47],
     ...
    [47, 12, 89]
    ]
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
                paquet.append(accs[i])
            # Changement d'activité
            else:
                paquet = [accs[i]]
                en_cours = acts[i]

        # Ajout du paquet
        else:
            if len(paquet) != PAQUET:
                print("error")
            paquet = np.array(paquet)
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

    outfile = './datas_keras.npz'
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
