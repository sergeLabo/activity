#!python3

import numpy as np
# #from tensorflow import keras
from tensorflow.keras import Sequential, layers


epochs = 5

def main():

    # #resp = []
    train, test, train_label, test_label = get_train_test_datas()
    print("11", train[11], train_label[11], test[11], test_label[11])

    model = build_the_model()
    model = compile_the_model(model)
    model = training_the_model(model, train, train_label, epochs)
    # #model.save('acc_model.h5')
    test_acc = testing_the_model(model, test, test_label)
    print("    Efficacité: ", round(test_acc*100, 1), "%")


def get_train_test_datas():

    print("Init ...")
    data = np.load('./npz_final/datas_keras.npz', allow_pickle=True)
    train = data["train"]
    test = data["test"]
    train_label = data["train_label"]
    test_label = data["test_label"]

    print("Taille:", train.shape, test.shape, train_label.shape, test_label.shape)

    return train, test, train_label, test_label

def build_the_model():
    print("\n\n\nBuild the model ...")

    model = Sequential()

    # Input layer
    model.add(layers.Dense(units=4, input_shape=(50, 3)))
    model.add(layers.Flatten())

    # Hiiden layer
    model.add(layers.Dense(64))


    # Output
    model.add(layers.Dense(7))

    print(model.summary())
    print("Build done.")

    return model

def compile_the_model(model):
    print("\n\n\nCompile the model ...")

    model.compile(  optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'] )

    print("Compile done.")
    return model

def training_the_model(model, train, train_label, epochs):

    print("\n\n\nTraining the model ...")
    print("Taille:", train.shape, train_label.shape)

    model.fit(train, train_label, epochs=epochs)

    print("Training done.")
    return model

def testing_the_model(model, test, test_label):
    print("\n\n\nTesting ......")
    test_loss, test_acc = model.evaluate(test, test_label)
    return test_acc

if __name__ == "__main__":
    main()
