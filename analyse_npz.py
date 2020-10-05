#!/usr/bin/env python3

import numpy as np


def acc_compare(num):
    data = np.load('./npz/' + str(num) + '.npz')
    x = data['x']
    y = data['y']
    z = data['z']  # array
    activity = data['activity']
    print("11", x[11], y[11], z[11])

    acc = []
    for i in range(len(x)):
        acc.append((x[i]**2 + y[i]**2 + z[i]**2 )**0.5)
    moy = np.average(acc)
    print(f"Numéro: {num}\n  Accélération moyenne = {int(moy)}")

    xm = np.average(x)
    ym = np.average(y)
    zm = np.average(z)
    print(f"    Accélération moyenne en x = {int(xm)}")
    print(f"    Accélération moyenne en y = {int(ym)}")
    print(f"    Accélération moyenne en z = {int(zm)}")

    return x, y, z, xm, ym, zm


def main():
    correction = []
    moyenne = [[], [], []]
    for num in range(1, 16, 1):
        x, y, z, xm, ym, zm = acc_compare(num)

        # Pour calcul de la moyenne globale
        for item in x: moyenne[0].append(item)
        for item in y: moyenne[1].append(item)
        for item in z: moyenne[2].append(item)

        # Table de correction
        # si xm trop petit, il faut ajouter correction[][] à tout x
        correction.append((1987 - xm, 2382 - ym, 1970 - zm))


    moyenne_x = int(sum(moyenne[0]) / len(moyenne[0]))
    moyenne_y = int(sum(moyenne[1]) / len(moyenne[1]))
    moyenne_z = int(sum(moyenne[2]) / len(moyenne[2]))

    print(moyenne_x,moyenne_y,moyenne_z)
    # 1987 2382 1970

    print(f"Correction: {correction}")


if __name__ == "__main__":
    main()



"""
Correction = [
(76.32921230769239, 1.7136738461540517, -71.21460307692314),
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

"""
