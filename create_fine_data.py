#!python3


import os
from time import time, sleep
import json
import ast
import numpy as np


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

def get_parts_data():
    """datas = liste des datas de chaque geek
    datas[0] = liste des x,y,z,act
    """
    pass

def main():

    global CONFIG

    datas = get_data_in_all_npz()
    print(f"Type de datas: {type(datas)} avec len() = {len(datas)}")
    # Type de datas: <class 'list'> avec len() = 15

    print(type(datas[0]), len(datas[0]))
    # <class 'list'> 4

    get_parts_data(datas)


if __name__ == "__main__":
    main()
