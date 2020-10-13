#!python3

import numpy as np
import random
from datetime import datetime
# #from tensorflow import keras
from tensorflow.keras import Sequential, layers

def main():

    resp = []

    for i in range(1):
        for PAQUET in [51]:  # [25, 51, 75, 101, 125, 151, 251, 301]:
            for window in [51]:  #  [21, 31, 41, 51, 61, 71, 81]:  # toujours impair
                for polyorder in [3]:  #  [1, 3, 5, 7, 9]:
                    for epochs in [5, 10, 20]:
                        train, test, train_label, test_label = get_train_test_datas(PAQUET=PAQUET,
                                                                                    window=window,
                                                                                    polyorder=polyorder)
                        model = build_the_model(PAQUET=PAQUET)
                        model = compile_the_model(model)
                        model = training_the_model(model, train, train_label, epochs=epochs)
                        # #model.save('acc_model.h5')
                        test_acc = testing_the_model(model, test, test_label)
                        a = "Paquets={} window={} polyorder={} Efficacité={}% indice={} Epochs={} "
                        print(a.format(PAQUET, window, polyorder, round(test_acc*100, 1), i, epochs))
                        resp.append([PAQUET, window, polyorder, test_acc, i, epochs])

    now = datetime.now()
    time_ = now.strftime('%Y_%m_%d_%H_%M')
    outfile = './hyperparameter/hyperparameter_2_' + time_ + '.npz'
    print("Save:", outfile)
    np.savez_compressed(outfile, resp)
    print("\n\n\n\n")
    for r in resp:
        print(a.format(r[0], r[1], r[2], round(r[3]*100, 1), r[4], r[5]))

def get_train_test_datas(**kwargs):

    PAQUET = kwargs.get('PAQUET', None)
    window = kwargs.get('window', None)
    polyorder = kwargs.get('polyorder', None)

    print("\n\n\n\nInit ... Chargement des datas")
    infile =    './npz_final/hyperparameter/smooth_numpy_' +\
                str(PAQUET) + '_' +\
                str(window) + '_' +\
                str(polyorder) +\
                '.npz'

    data = np.load(infile, allow_pickle=True)
    train = data["train"]
    test = data["test"]
    train_label = data["train_label"]
    test_label = data["test_label"]

    print("Taille:", train.shape, test.shape, train_label.shape,
                    test_label.shape)

    return train, test, train_label, test_label

def build_the_model(**kwargs):
    print("Build the model ...")

    PAQUET = kwargs.get('PAQUET', None)

    # Choix du model
    model = Sequential()

    # Input layer
    model.add(layers.Dense(units=4, input_shape=(PAQUET, 3)))
    model.add(layers.Flatten())

    # Hiiden layer
    model.add(layers.Dense(64))
    # #model.add(layers.Dense(64))

    # Output
    model.add(layers.Dense(7))

    print(model.summary())
    print("Build done.")

    return model

def compile_the_model(model, **kwargs):

    print("Compile the model ...")
    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'] )

    print("Compile done.")
    return model

def training_the_model(model, train, train_label, **kwargs):

    epochs = kwargs.get('epochs', None)

    print("Training the model ...")
    model.fit(train, train_label, epochs=epochs)
    print("Training done.")

    return model

def testing_the_model(model, test, test_label, **kwargs):

    print("Testing ......")
    test_loss, test_acc = model.evaluate(test, test_label)
    return test_acc

if __name__ == "__main__":
    main()
