#!python3


import os
from time import time, sleep
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


PAQUET = 50


def load_config():
    global CONFIG
    fichier = './config.json'
    with open(fichier, "r") as fd:
        conf = fd.read()
        fd.close()
    return ast.literal_eval(conf)

CONFIG = load_config()

def get_data_in_all_npz():
    """len de datas = 15 datas[i] = [x, y, z, activity]"""

    datas = []
    for geek in range(1, 16, 1):
        [x, y, z, activity] = get_data_per_npz(geek)
        datas.append([x, y, z, activity])
    print(f"Nombre d'humains = {len(datas)}")
    return datas

def get_data_per_npz(geek):
    data = np.load('./npz/' + str(geek) + '.npz')
    x = data['x']
    y = data['y']
    z = data['z']  # array
    activity = data['activity']

    return [x, y, z, activity]

def get_only_one_activity(datas, activ, geek):
    """Activity = 3, c'est index 2 ???"""

    x, y, z, activity = datas[geek-1]

    acc = [[]]
    en_cours = 0
    n = 0
    for i in range(x.shape[0]):
        # Activity = 3, c'est index 2
        if activity[i] == activ - 1:
            #a = int((x[i]**2 + y[i]**2 + z[i]**2)**0.5)
            acc[n].append([x[i], y[i], z[i]])

        elif activity[i] != en_cours:
            acc.append([])
            n += 1
            en_cours = activity[i]

    # Suppression des listes vides
    acc_clean = [x for x in acc if x]
    acc = acc_clean
    print(f"Nombre de groupes de type {activ} = {len(acc)}" )
    i = 0
    for item in acc:
        print(f"    Group n°{i} nombre de valeurs = {len(item)}")
        i += 1
    return acc

def get_train_test_datas(datas):
    """
    train_test       = [[paquet de 50*[12,14,16] de la même activité], .... ]
    train_test_label = [2=activity, .... ]

    CONFIG[str(activ)][str(geek)]['mini_' + str(gr)] = int(mini)
    CONFIG[str(activ)][str(geek)]['maxi_' + str(gr)] = int(maxi)
    CONFIG = {'1=': {'1': {'mini_0': 572, 'maxi_0': 32531}, '2': {'mini_0': 2958,
    'maxi_0': 44150}, '3': {'mini_0': 916, 'maxi_0': 41675},

    """
    global CONFIG

    train_test, train_test_label = [], []

    print(f"Get all datas ...")
    datas = get_data_in_all_npz()
    print(f"Done.\n\n")
    for activ in range(1, 8, 1):  # 7 valeurs de 1 à 7
        for geek in range(1, 16, 1):  # 15 humains
            # list de tous les acc pour activ et geek
            print(f"Get datas of geek: {geek} and activity: {activ}")
            accs = get_only_one_activity(datas, activ, geek)
            print(f"Nombre de groupe: {len(accs)}\n")

            # Parcours des groupe
            gr = 0
            for groupe in accs:
                print(f"Plot ... activity={activ} geek={geek} gr={gr}")
                nb = len(groupe)
                print(f"     Nombre de valeurs: {nb}")

                debut = CONFIG[str(activ)][str(geek)]['mini_' + str(gr)]
                fin = CONFIG[str(activ)][str(geek)]['maxi_' + str(gr)]

                bon = groupe[debut:fin]
                print(debut, fin, len(groupe), "Reste:", len(bon))

                # Pour créer l'axe des x
                x_values = [a for a in range(len(bon))]
                # les y
                yx = [b[0] for b in bon]
                yy = [b[1] for b in bon]
                yz = [b[2] for b in bon]
                fig, ax = plt.subplots(figsize=(20,10), facecolor='#cccccc')
                timer = fig.canvas.new_timer(interval = 200)
                timer.add_callback(plt.close)
                plt.subplots_adjust(left=0.1, bottom=0.15)
                ax.set_facecolor('#eafff5')
                ax.set_title("Activity " + str(activ) + " Geek " + str(geek) + " Groupe " + str(gr),
                                size=24, color='magenta')
                ax.set_ylim(1000, 4000)
                plt.plot(x_values, yx, color='r', marker='X', linewidth=0.02)
                plt.plot(x_values, yy, color='g', marker = 'X', linewidth=0.02)
                plt.plot(x_values, yz, color='b', marker = 'X', linewidth=0.02)

                paquets = get_paquets(bon)
                for paquet in paquets:
                    train_test.append(paquet)
                    train_test_label.append(activ)

                # #timer.start()
                # #plt.show()
                gr += 1

    return train_test, train_test_label

def get_paquets(bon):
    nb = len(bon)
    nb_paquet = int(nb/PAQUET)
    paquets = [0]*nb_paquet
    print(f"Nombre de paquet = {nb_paquet}")
    for i in range(nb_paquet):
        for item in bon:
            paquets[i] = item
    return paquets

def create_array(train_test, train_test_label):
    # je bats les cartes des datas, mais conserve la correspondance valeur, label
    par_couple = {}
    p = 0
    for p in range(len(train_test)):
        par_couple[p] = (train_test[p], train_test_label[p])

    train = []
    train_label = []

    n_train = int(nb_paquet*0.8)
    n_test = nb_paquet - n_train
    print(f"Nombre de valeurs pour le training = {n_train}")
    print(f"Nombre de valeurs pour le testing = {n_test}")

    # liste de nombre au hasard qui seront les indices
    hasard = [x for x in range(len(par_couple))]
    random.shuffle(hasard)
    for item in hasard[:n_train]:
        train.append(par_couple[item][0])
        train_label.append(par_couple[item][1])
    for item in hasard[n_train:]:
        train.append(par_couple[item][0])
        train_label.append(par_couple[item][1])

    train = np.array(train_test)
    test  = np.array(train_test)
    train_label = np.array(train_test_label)
    test_label = np.array(train_test_label)

    print("Taille", train.shape, test.shape, train_label.shape, test_label.shape)
    return train, test, train_label, test_label

def save_npz(train, test, train_label, test_label):
    outfile = './clean_numpy.npz'
    np.savez_compressed(outfile, **{"train": train,
                                    "test": test,
                                    "train_label": train_label,
                                    "test_label":  test_label})
    print('Fichier compressé =', outfile)


def main():

    datas = get_data_in_all_npz()
    print(f"Type de datas: {type(datas)} avec len() = {len(datas)}")

    train_test, train_test_label = get_train_test_datas(datas)
    train, test, train_label, test_label = create_array(train_test, train_test_label)
    save_npz(train, test, train_label, test_label)



if __name__ == "__main__":
    main()
