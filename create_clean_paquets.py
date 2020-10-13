#!python3


import os
from time import time, sleep
import random
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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
    print(f"Nombre datas pour geek {geek} : {len(x)}")
    return [x, y, z, activity]

def get_only_one_activity(datas, activ, geek):
    """Activity = 3, c'est index 2 ???"""

    x, y, z, activity = datas[geek-1]

    accs = [[]]
    en_cours = 0
    n = 0
    for i in range(x.shape[0]):
        # Activity = 3, c'est index 2
        if activity[i] == activ - 1:
            accs[n].append([x[i], y[i], z[i]])

        elif activity[i] != en_cours:
            accs.append([])
            n += 1
            en_cours = activity[i]

    # Suppression des listes vides
    accs = [x for x in accs if x]
    print(f"        Nombre de groupes de type {activ} = {len(accs)}" )
    i = 0
    for item in accs:
        print(f"        Group n°{i} nombre de valeurs = {len(item)}")
        i += 1

    # accs = [[[1200, 800, 600], ...], [2 ème groupe]] pour activité activ
    return accs

def get_train_test_datas_and_plot(datas, PAQUET):
    """
    train_test       = [[paquet de 50*[12,14,16] de la même activité], .... ]
    train_test_label = [2=activity, .... ]

    CONFIG[str(activ)][str(geek)]['mini_' + str(gr)] = int(mini)
    CONFIG[str(activ)][str(geek)]['maxi_' + str(gr)] = int(maxi)
    CONFIG = {'1=': {'1': {'mini_0': 572, 'maxi_0': 32531}, '2': {'mini_0': 2958,
    'maxi_0': 44150}, '3': {'mini_0': 916, 'maxi_0': 41675},

    """
    global CONFIG
    plot_ = 0

    train_test, train_test_label = [], []

    for activ in range(1, 8, 1):  # 7 valeurs de 1 à 7
        for geek in range(1, 16, 1):  # 15 humains
            # list de tous les acc pour activ et geek
            print(f"\n    Get datas of geek: {geek} and activity: {activ}")
            # acc = [[[1200, 800, 600], ...], [2 ème groupe]] pour activité activ
            accs = get_only_one_activity(datas, activ, geek)
            print(f"        Nombre de groupe: {len(accs)}")

            # Parcours des groupes
            gr = 0
            for groupe in accs:
                print(f"        Analyse de activity={activ} geek={geek} gr={gr}")
                nb = len(groupe)
                print(f"        Nombre de valeurs: {nb}")

                debut = CONFIG[str(activ)][str(geek)]['mini_' + str(gr)]
                fin = CONFIG[str(activ)][str(geek)]['maxi_' + str(gr)]

                if fin <= debut:
                    sp = " " * 40
                    print(f"{sp} erreur mini > maxi", debut, fin, activ, geek, gr)

                good_datas = groupe[debut:fin]
                print("        Début:", debut, "Fin:", fin, "nombre de val:",len(groupe),
                                "Reste:", len(good_datas))

                if plot_:
                    # Pour créer l'axe des x
                    x_values = [a for a in range(len(good_datas))]
                    # les y
                    yx = [b[0] for b in good_datas]
                    yy = [b[1] for b in good_datas]
                    yz = [b[2] for b in good_datas]
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
                    timer.start()
                    plt.show()

                paquets = get_paquets(good_datas, PAQUET)
                for paquet in paquets:
                    # #print("taoazo", len(paquet))
                    train_test.append(paquet)
                    train_test_label.append(activ-1)

                gr += 1

    return train_test, train_test_label

def get_paquets(good_datas, PAQUET):
    """Si paquet de 3
    [[], [], [], [], [], [], [], [], ..
    devient
    [[[], [], []], 3 list dans le paquet 1 i = 0 p de 0 à 50: j=0  à 49
     [[], [], []], 3 list dans le paquet 2 i = 1 p de 0 à 50: j=50 à 99
     [[], ...
    """

    nb = len(good_datas)
    nb_paquet = int(nb/PAQUET)

    # paquet = [ [ 50*3 ], [ 50*3 ] ...]
    # #paquets = [0]*nb_paquet

    paquets = [[]]
    print(f"        Nombre de paquet à créer = {nb_paquet}")

    p = 0 # indexation des paquets
    t = 0 # indexation des triplets dans le paquet
    for triplet in good_datas:
        if p <= nb_paquet:
            if t < PAQUET: # la création à déjà ajouté un paquet, il faut en ajouter 49
                paquets[p].append(triplet)
                t += 1
            else:
                p += 1
                if p < nb_paquet:  # le paquet PAQUET serait incomplet
                    paquets.append([triplet])
                    t = 1
                else:
                    p = nb_paquet + 1

    print("        Vérification du dernier paquet", len(paquets[-1]))
    print("        Nombre de paquets réel:", len(paquets))
    return paquets

def shuffle(train_test, train_test_label):

    # Je vérifie
    print("Taille de train_test:", len(train_test))
    print(len(train_test[0]))
    print(len(train_test[0][0]))

    # je bats les cartes des datas, mais conserve la correspondance valeur, label
    par_couple = {}
    p = 0
    for p in range(len(train_test)):
        par_couple[p] = (train_test[p], train_test_label[p])
    train, test, train_label, test_label = [], [], [], []

    total = len(train_test)
    print(f"Nombre de paquets dans train et test = {total}")
    n_train = int(total*0.8)
    n_test = total - n_train
    print(f"Nombre de valeurs pour le training = {n_train}")
    print(f"Nombre de valeurs pour le testing = {n_test}")

    # liste de nombre au hasard qui seront les indices
    hasard = [x for x in range(len(par_couple))]
    random.shuffle(hasard)

    print("Remplissage des listes train test labels ...")
    for item in hasard[:n_train]:
        train.append(par_couple[item][0])
        train_label.append(par_couple[item][1])
    for item in hasard[n_train:]:
        test.append(par_couple[item][0])
        test_label.append(par_couple[item][1])

    return train, test, train_label, test_label

def save_npz(train, test, train_label, test_label, PAQUET):

    print("Vérification avant enregistrement:")
    print("    ", train.shape, test.shape, train_label.shape, test_label.shape)

    outfile = './npz_final/clean_numpy_' + str(PAQUET) + '.npz'
    np.savez_compressed(outfile, **{"train": train,
                                    "test": test,
                                    "train_label": train_label,
                                    "test_label":  test_label})
    print('Fichier compressé =', outfile)

def create_arrays(train, test, train_label, test_label, PAQUET):

    # le facile
    train_label_a = np.array(train_label)
    test_label_a = np.array(test_label)
    print("Shape des labels", train_label_a.shape, test_label_a.shape)

    train_a = create_array(train, PAQUET)
    test_a = create_array(test, PAQUET)

    return train_a, test_a, train_label_a, test_label_a

def create_array(train, PAQUET):
    """
    train = [30 000 items soit des groupes : 30000 = len(train)
                [ 50 items soit des paquets
                    [ 3 items soit un triplet

    train_a.shape = (30000, 50, 3)
    """

    train_a = np.zeros((len(train), PAQUET, 3), np.uint8)
    for g in range(len(train)):
        for p in range(PAQUET):
            for t in range(3):
                train_a[g][p][t] = train[g][p][t]
    return train_a

def main():

    print(f"\n\nRécupération des toutes les datas dans tous les npz ...")
    datas = get_data_in_all_npz()
    print(f"\nType de datas: {type(datas)} avec len() = {len(datas)}")

    for PAQUET in [25, 50, 75, 100, 125, 150]:
        print("\nGet_train_test_datas_and_plot ...")
        train_test, train_test_label = get_train_test_datas_and_plot(datas, PAQUET)

        print("\nShuffle avec hasard ...")
        train, test, train_label, test_label = shuffle(train_test, train_test_label)

        print("\nCreation des array avec hasard ...")
        train, test, train_label, test_label = create_arrays(train, test, train_label, test_label, PAQUET)

        print("\nSave npz ...")
        save_npz(train, test, train_label, test_label, PAQUET)
        print("Done.")


if __name__ == "__main__":
    main()


"""
for num in range(1, 16, 1):
    x, y, z, activity =  tout est array
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
"""
