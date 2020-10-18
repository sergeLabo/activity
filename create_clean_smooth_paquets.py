#!python3

"""
Crée les ./npz_final/hyperparameter/no_smooth_keras_25_21_1.npz
Ne garde que les bonnes datas
smooth ou pas
plot ou pas
"""


import random
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.signal


def main():

    kwargs = {  "PAQUET": 900,
                "window": 61, # impair
                "polyorder": 3,
                "save": 1,  # pour faire enreg
                "plot": 1,  # pour afficher les courbes
                "smooth": 1,  # lissage
                "dt": 3000,  # ms d'affichage
                "gliss": 100,  # paquets glissants
                "clean": 1,  # coupe des début fin d'activité
                "fullscreen": 1
             }

    gttn = getTrainTestNpz(**kwargs)


class getTrainTestNpz:

    def __init__(self, **kwargs):

        self.PAQUET = kwargs.get('PAQUET', None)
        self.window = kwargs.get('window', None)
        self.polyorder = kwargs.get('polyorder', None)
        self.smooth = kwargs.get('smooth', None)
        self.plot = kwargs.get('plot', None)
        self.save = kwargs.get('save', None)
        self.dt = kwargs.get('dt', None)
        self.gliss = kwargs.get('gliss', None)
        self.clean = kwargs.get('clean', None)
        self.fullscreen = kwargs.get('fullscreen', None)
        self.config = load_config()

        print(f"\n\nRécupération des toutes les datas dans tous les npz ...")
        self.datas = self.get_data_in_all_npz()

        print("\nGet_train_test_datas_and_plot ...")
        train_test, train_test_label = self.get_train_test_datas_and_plot()
        print("\nShuffle et to_array ...")
        train, test, train_label, test_label = self.shuffle(train_test, train_test_label)
        print("\nSave npz ...")
        if self.save:
            self.save_npz(train, test, train_label, test_label)
        print("Done.")

    def get_data_in_all_npz(self):
        """len de datas = 15 datas[i] = [x, y, z, activity]"""

        datas = []
        for geek in range(1, 16, 1):
            datas.append(self.get_data_per_npz(geek))
        return datas

    def get_data_per_npz(self, geek):

        data = np.load('./npz/' + str(geek) + '.npz')
        if self.smooth:
            x = scipy.signal.savgol_filter(data['x'], self.window, self.polyorder)
            y = scipy.signal.savgol_filter(data['y'], self.window, self.polyorder)
            z = scipy.signal.savgol_filter(data['z'], self.window, self.polyorder)  # array
        else:
            x = data['x']
            y = data['y']
            z = data['z']  # array
        activity = data['activity']

        return [x, y, z, activity]

    def get_only_one_activity(self, activ, geek):
        """Activity = 3, c'est index 2 ???"""

        x, y, z, activity = self.datas[geek-1]

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
        i = 0
        for item in accs:
            i += 1

        return accs

    def get_train_test_datas_and_plot(self):
        """
        train_test       = [[paquet de 50*[12,14,16] de la même activité], .... ]
        train_test_label = [2=activity, .... ]
        CONFIG = {'1=': {'1': {'mini_0': 572, 'maxi_0': 32531}, '2': {'mini_0': 2958,
        'maxi_0': 44150}, '3': {'mini_0': 916, 'maxi_0': 41675},
        """

        train_test, train_test_label = [], []

        for activ in range(1, 8, 1):  # 7 valeurs de 1 à 7
            for geek in range(1, 16, 1):  # 15 humains
                # list de tous les acc pour activ et geek
                # acc = [[[1200, 800, 600], ...], [2 ème groupe]] pour activité activ
                accs = self.get_only_one_activity(activ, geek)

                # Parcours des groupes
                gr = 0
                for groupe in accs:
                    nb = len(groupe)

                    if self.clean:
                        debut = self.config[str(activ)][str(geek)]['mini_' + str(gr)]
                        fin = self.config[str(activ)][str(geek)]['maxi_' + str(gr)]
                        good_datas = groupe[debut:fin]
                    else:
                        good_datas = groupe

                    if self.plot:
                        # Pour créer l'axe des x
                        x_values = [a for a in range(len(good_datas))]
                        # les y
                        yx = [b[0] for b in good_datas]
                        yy = [b[1] for b in good_datas]
                        yz = [b[2] for b in good_datas]

                        fig, ax = plt.subplots(figsize=(10,10), facecolor='#cccccc')
                        if self.fullscreen:
                            fig.canvas.manager.full_screen_toggle()
                        timer = fig.canvas.new_timer(interval = self.dt)
                        timer.add_callback(plt.close)
                        plt.subplots_adjust(left=0.1, bottom=0.15)
                        ax.set_facecolor('#eafff5')
                        ax.set_title("Activity " + str(activ) + " Geek " + str(geek) +\
                                     " Groupe " + str(gr), size=24, color='magenta')
                        ax.set_ylim(1000, 4000)

                        # Création des courbes
                        plt.plot(x_values, yx, color='r', marker='X', linewidth=0.02)
                        plt.plot(x_values, yy, color='g', marker = 'X', linewidth=0.02)
                        plt.plot(x_values, yz, color='b', marker = 'X', linewidth=0.02)

                        timer.start()
                        if self.smooth:
                            f = "./courbe/zones_final_smooth/activ_" +\
                                 str(activ) + "_geek_" + str(geek) + "_groupe_" +\
                                  str(gr) + "_smooth_" + str(self.smooth) + ".png"
                        else:
                            f = "./courbe/zones_final_avant_smooth/activ_" +\
                                str(activ) + "_geek_" + str(geek) + "_groupe_" +\
                                 str(gr) + ".png"
                        fig.savefig(f)
                        plt.show()

                    paquets = self.get_paquets_gliss(good_datas)

                    for paquet in paquets:
                        train_test.append(paquet)
                        train_test_label.append(activ-1)

                    gr += 1

        return train_test, train_test_label

    def get_paquets_gliss(self, good_datas):
        """ Exemple avec PAQUET=1000, len(datas) = 4954, gliss=10
            nombre de paquet possible = int((4954-1000)/10) = 395
            395*10 + 1000 = 3950+1000= 4950 perte de 4! seulement
            le paquet 0 commence à datas[0] fini à datas[999]
            le paquet 1 commence à datas[10] fini à datas[1099]
            jusqu'à nb_possible commence à datas[???] fini à datas[nb_possible*1000]
        """

        nb_possible = int((len(good_datas) - self.PAQUET)/self.gliss)
        e = " "*10

        paquets = []
        if nb_possible > 0:
            # p = indexation des paquets = commence à 0, fini à nb_possible
            # t = indexation des triplets dans le paquet
            for p in range(nb_possible):
                paquet = []
                for t in range(self.PAQUET):
                    # p=0 et t=0à999 --> 0 1 2 3   ... 999
                    # p=1 et t=0à999 --> 10 11 12 .... 1009
                    paquet.append(good_datas[self.gliss*p + t])
                paquets.append(paquet)
        # #print("        Nombre de paquets réel:", len(paquets))
        return paquets

    def shuffle(self, train_test, train_test_label):
        """Le shuffle se fait en conservant la correspondance entre l'indice
        du paquet(50,3) et son label.
        Je convertit en array à la fin
        """

        # Je vérifie
        print("Taille de train_test:", len(train_test))

        # je bats les cartes des datas, mais conserve la correspondance valeur, label
        par_couple = {}
        p = 0
        for p in range(len(train_test)):
            par_couple[p] = (train_test[p], train_test_label[p])
        train, test, train_label, test_label = [], [], [], []

        total = len(train_test)
        n_train = int(total*0.8)
        n_test = total - n_train

        # liste de nombre au hasard qui seront les indices
        hasard = [x for x in range(len(par_couple))]
        random.shuffle(hasard)

        for item in hasard[:n_train]:
            train.append(par_couple[item][0])
            train_label.append(par_couple[item][1])
        for item in hasard[n_train:]:
            test.append(par_couple[item][0])
            test_label.append(par_couple[item][1])

        # Conversion en array
        train = np.array(train)
        test = np.array(test)
        train_label = np.array(train_label)
        test_label = np.array(test_label)

        return train, test, train_label, test_label

    def save_npz(self, train, test, train_label, test_label):

        if self.save:
            print("Vérification avant enregistrement:")
            print("    ", train.shape, test.shape, train_label.shape, test_label.shape)

            outfile = (f'./npz_final/hyperparameter/keras_{self.PAQUET}_'
                       f'{self.window}_{self.polyorder}_{self.gliss}_'
                       f'{self.smooth}_{self.clean}.npz')

            print('Fichier compressé =', outfile)

            np.savez_compressed(outfile, **{"train": train,
                                            "test": test,
                                            "train_label": train_label,
                                            "test_label":  test_label})


def load_config():
    fichier = './config.json'
    with open(fichier, "r") as fd:
        conf = fd.read()
        fd.close()
    return ast.literal_eval(conf)


if __name__ == "__main__":
    main()
