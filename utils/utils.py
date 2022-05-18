import pickle
import json

def save(obj,path_name):
    try:
        with open(path_name,'wb') as file:
            print("save to:",path_name)
            pickle.dump(obj,file)
    except  IOError as ioerror:
        print(ioerror)
        print("Error when saving: {}".format(path_name))

def load(path_name: object) -> object:
    try:
        with open(path_name,'rb') as file:
            return pickle.load(file)
    except IOError as ioerror:
        print(ioerror)
        print("Error when loading: {}".format(path_name))

def loadJson(path_name:str):
    try:
        with open(path_name,'rb') as file:
            data = json.load(file)
            return data
    except IOError as ioerror:
        print(ioerror)
        print("Error when loading: {}".format(path_name))