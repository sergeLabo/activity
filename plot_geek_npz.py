#!python3

"""
plot les courbes de chaque npz originaux
"""


import numpy as np
import matplotlib.pyplot as plt


def plot(num):
    data = np.load('./npz/' + str(num) + '.npz')
    x = data['x']
    y = data['y']
    z = data['z']  # array
    activity = data['activity']

    acc = []
    for i in range(len(x)):
        acc.append((x[i]**2 + y[i]**2 + z[i]**2 )**0.5)

    # Pour créer l'axe des x
    x_values = [a for a in range(len(x))]

    fig, ax1 = plt.subplots(1, 1, figsize=(10,10), facecolor='#cccccc')
    ax1.set_facecolor('#eafff5')
    ax1.set_title("Activity " + str(num) , size=24, color='magenta')

    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Accélération', color=color)
    a = ax1.scatter(x_values, acc,
                    marker = 'X',
                    # #linestyle="-",
                    linewidth=0.05,
                    color='green',
                    label="Accélération")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Activity', color='tab:red')  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)

    b = ax2.plot(x_values, activity,
                linestyle="-",
                linewidth=1.5,
                color='red',
                label="Activity")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(loc="upper center")
    ax2.legend(loc="upper right")

    fig.savefig("./courbe/activity_raw/activity_" + str(num) + ".png")
    plt.show()


def main():
    for num in range(1, 16, 1):
        plot(num)

if __name__ == "__main__":
    main()
