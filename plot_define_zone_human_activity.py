#!python3

"""
Définition des zones conservés pour chaque zone d'activité par geek
Construit le json des bornes
Si relancé, affiche les bornes tirées du json, elles sont modifiables
"""


import os
from time import time, sleep
import json
import ast
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import matplotlib
print(f"Matplotlib: Version = {matplotlib.__version__}")
print(f"Numpy: Version = {np.__version__}")

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
            a = int((x[i]**2 + y[i]**2 + z[i]**2)**0.5)
            acc[n].append(a)

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

def plot_geek_activities(datas, activ, geek):
    """len de datas = 15 datas[i] = [x, y, z, activity]
    './activity.ini'
    """

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

        # Pour créer l'axe des x
        x_values = [a for a in range(nb)]
        # les y
        y_values = groupe
        fig, ax = plt.subplots(figsize=(20,10), facecolor='#cccccc')
        plt.subplots_adjust(left=0.1, bottom=0.15)
        ax.set_facecolor('#eafff5')
        ax.set_title("Activity " + str(activ) + " Geek " + str(geek) + " Groupe " + str(gr),
                        size=24,
                        color='magenta')
        ax.set_ylim(3000, 5000)

        l = plt.plot(x_values, y_values, marker = 'X', linewidth=0.02)

        mini = CONFIG[str(activ)][str(geek)]["mini_" + str(gr)]
        maxi = CONFIG[str(activ)][str(geek)]["maxi_" + str(gr)]
        vlm = ax.axvline(mini)
        vlM = ax.axvline(maxi)

        # Slider
        axmini = plt.axes([0.25, .02, 0.50, 0.02]) # position, taille
        slidmin = Slider(axmini, 'Mini', 0, nb, valinit=mini)
        axmaxi = plt.axes([0.25, .05, 0.50, 0.02])
        slidmax = Slider(axmaxi, 'Maxi', 0, nb, valinit=maxi)

        def update_mini(mini):
            nonlocal vlm
            mini = slidmin.val
            vlm.remove()
            # update curve
            vlm = ax.axvline(mini)
            CONFIG[str(activ)][str(geek)]["mini_" + str(gr)] = int(mini)

        def update_maxi(maxi):
            nonlocal vlM
            maxi = slidmax.val
            vlM.remove()
            # update curve
            vlM = ax.axvline(maxi)
            CONFIG[str(activ)][str(geek)]["maxi_" + str(gr)] = int(maxi)

        # call update function on slider value change
        slidmin.on_changed(update_mini)
        slidmax.on_changed(update_maxi)

        fig.savefig("./courbe/zone_defined/activ_" + str(activ) + "_geek_" + str(geek) + "_groupe_" + str(gr) + ".png")
        plt.show()
        gr += 1

def save_config():
    global CONFIG
    conf = json.dumps(CONFIG)
    fichier = './config.json'
    with open(fichier, "w") as fd:
        fd.write(conf)
        fd.close()

def load_config():
    global CONFIG
    fichier = './config.json'
    with open(fichier, "r") as fd:
        conf = fd.read()
        fd.close()
    return ast.literal_eval(conf)

CONFIG = load_config()

def main():
    global CONFIG
    print(f"Get all datas ...")
    datas = get_data_in_all_npz()
    print(f"Done.\n\n")
    for activ in range(1, 8, 1):  # 7 valeurs de 1 à 7
        for geek in range(1, 16, 1):  # 15 humains
            plot_geek_activities(datas, activ, geek)
            save_config()

    print(CONFIG)

if __name__ == "__main__":
    main()
