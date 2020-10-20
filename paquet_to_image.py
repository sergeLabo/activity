#!python3

"""
Création d'images
"""


import numpy as np
# #print(np.version.version)
import cv2
import sounddevice as sd

def main():
    pti = PaquetToImage()

class PaquetToImage:

    def __init__(self, **kwargs):

        infile = './npz_final/hyperparameter/keras_900_61_3_100_1_1.npz'
        data = np.load(infile, allow_pickle=True)
        print(f"Chargement de {infile} ...")
        self.get_train_test_datas(data)
        print(f"Vérification des shapes: {self.train.shape} {self.test.shape} {self.train_label.shape} {self.test_label.shape}")

        self.black_image = np.zeros((900, 3), np.uint8)
        self.display()

    def get_train_test_datas(self, data):

        self.train = data["train"]
        self.test = data["test"]
        self.train_label = data["train_label"]
        self.test_label  = data["test_label"]

    def display(self):

        i = -1
        while i < self.train.shape[0] - 1:
            i += 1
            # #print(self.train[0][0])
            black_image = self.black_image.copy()

            # shape = (900, 3) to ((30, 30, 3)
            img = np.reshape(self.train[i], (30, 30, 3))
            img = img//25
            img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

            alpha = 5 # Contrast control (1.0-3.0)
            beta = 1 # Brightness control (0-100)
            # #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

            img = cv2.resize(img, (900, 900), interpolation=cv2.INTER_NEAREST)
            # Affichage
            if img.any():
                cv2.imshow('Activity', img)

            label = self.train_label[i]
            data = np.random.uniform(-1, 1, int(44100*label/7))
            sd.play(data, 44100)

            k = cv2.waitKey(10)
            if k == 27:
                break

        cv2.destroyAllWindows()


def improve_image(img):

    #-----Converting image to LAB Color model-----------------------------------
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(1, 1))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final


if __name__ == "__main__":

    main()
