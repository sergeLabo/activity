#!/usr/bin/env python3

import numpy as np


def acc_compare(num):
    data = np.load('./npz/' + str(num) + '.npz')
    x = data['x']
    y = data['y']
    z = data['z']  # array
    activity = data['activity']

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

def main():
    for num in range(1, 16, 1):
        acc_compare(num)


if __name__ == "__main__":
    main()
