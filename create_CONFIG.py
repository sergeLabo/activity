#!python3

"""
Script à lancer une fois pour initier le fichier config.json,
comme il existe dans le dépot, ne pas le lancer
"""


import json
import ast


CONFIG = {}

def save_config():
    global CONFIG
    conf = json.dumps(CONFIG)
    fichier = './config.json'
    with open(fichier, "w") as fd:
        fd.write(conf)
        fd.close()

def load_config():
    global CONFIG
    try:
        fichier = './config.json'
        with open(fichier) as fd:
            conf = fd.read()
            fd.close()
            print(f"config.json existe")
            return ast.literal_eval(conf)
    except IOError:
        print("config.json n'existe pas. Je le crée")
        init_CONFIG()
        save_config()

def init_CONFIG():
    """A utiliser pour créer une config initiale vide"""
    global CONFIG

    for activ in range(1, 8, 1):  # 7 valeurs de 1 à 7
        CONFIG[activ] = {}
        for geek in range(1, 16, 1):
            CONFIG[activ][geek] = {}
    print(f"Config initiale:")
    print(f"    {CONFIG}")

def main():
    print("Création de config.json si besoin")
    load_config()
    print(f"Done.\n\n")


if __name__ == "__main__":
    main()
