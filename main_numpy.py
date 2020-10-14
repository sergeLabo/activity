#!python3

"""
script joseph numpy avec des array input_shape=(50,)
sans nettoyage des datas et sans smooth
"""


import numpy as np


def sigmoid(x):
    try:
        a = 1 / (1 + np.exp(-x))
    except:
        a = 0.0
    return a
def sigmoid_prime(z): return z * (1 - z)
def relu(x): return np.maximum(0, x)
def relu_prime(z):return np.asarray(z > 0, dtype=np.float32)

PAQUET = 50
NB_OBJ = 7

class Activity:

    def __init__(self):
        print("Init ... avec ./datas_numpy.npz")
        data = np.load('./datas_numpy.npz', allow_pickle=True)
        self.train = data["train"]
        self.test = data["test"]
        self.train_label = data["train_label"]
        self.test_label = data["test_label"]

        print("11", self.train[11], self.train_label[11], self.test[11], self.test_label[11])
        print("Doit afficher 0:", self.train_label[0])

        self.layers = [PAQUET, 100, 100, NB_OBJ]
        self.activations = [relu, relu, sigmoid]
        self.diagonale = np.eye(NB_OBJ, NB_OBJ)
        self.activations_prime = [globals()[fonction.__name__ + '_prime'] for fonction in self.activations]
        self.weight_init = [np.random.randn(self.layers[k+1], self.layers[k]) / np.sqrt(self.layers[k]) for k in range(len(self.layers)-1)]

    def training(self, learningrate):
        print("Training ...")
        node_dict = {}
        weight_list = self.weight_init

        for i, (vecteur_ligne, activity) in enumerate(zip(self.train, self.train_label)):
            vecteur_colonne = np.array(np.array(vecteur_ligne), ndmin=2).T
            # vecteur_colonne doit être 2D (100, 1)
            # print(vecteur_colonne.shape)  # il faut avoir (100, 1)

            node_dict[0] = vecteur_colonne
            for k in range(len(self.layers)-1):
                z = np.dot(weight_list[k], vecteur_colonne)
                vecteur_colonne = self.activations[k](z)
                node_dict[k+1] = vecteur_colonne

            delta_a = vecteur_colonne - self.diagonale[:,[activity]]
            for k in range(len(self.layers)-2, -1, -1):
                delta_z = delta_a * self.activations_prime[k](node_dict[k+1])
                delta_w = np.dot(delta_z, node_dict[k].T)

                delta_a = np.dot(weight_list[k].T, delta_z)
                weight_list[k] -= learningrate * delta_w

        return weight_list

    def testing(self, weight_list):
        print("Testing ...")
        success = 0

        for vecteur_ligne, activity in zip(self.test, self.test_label):
            for k in range(len(self.layers)-1):
                vecteur_ligne = self.activations[k](np.dot(weight_list[k],
                                                      vecteur_ligne))
            reconnu = np.argmax(vecteur_ligne)
            if reconnu == activity:
                success += 1

        resp = 100.0 * success / len(self.test)

        return resp


if __name__ == "__main__":

    act = Activity()
    learningrate = 0.022
    weight_list = act.training(learningrate)
    resp = act.testing(weight_list)

    print(f"Learningrate: {learningrate} Résultat {round(resp, 1)}%")
