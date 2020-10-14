#!python3

"""
1er script à faire tourner pour convertir le csv de 50 Mo
en 15 npz compressé soit 6.7 Mo
"""


import os
import csv
import numpy as np


dossier = './Activity Recognition/'

def csv_to_list(csv_file):
    """0,1502,2215,2153,1"""

    with open(csv_file, newline='') as f:
        reader = csv.reader(f)

        x, y, z, a = [], [], [], []
        for row in reader:
            if int(row[4]) != 0:
                x.append(int(row[1]))
                y.append(int(row[2]))
                z.append(int(row[3]))
                a.append(int(row[4]) - 1) # de 0 à 6, au lieu de 1 à 7

    return x, y, z, a

def csv_to_npz():
    for i in range(1, 16, 1):
        x, y, z, a = csv_to_list(dossier + str(i) + '.csv')
        print("Première ligne", x[0], y[0], z[0], a[0])
        outfile = './npz/' + str(i) + '.npz'
        np.savez_compressed(outfile, **{"x": np.asarray(x),
                                        "y": np.asarray(y),
                                        "z": np.asarray(z),
                                        "activity": np.asarray(a)})
        print('Fichier compressé =', outfile)

def main():
    csv_to_npz()


if __name__ == "__main__":
    main()
