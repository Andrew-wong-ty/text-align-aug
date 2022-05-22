import pickle
import json
import random
import torch
import numpy as np
import transformers


def set_global_random_seed(seed):
    # set seed
    print("set seed:",seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save(obj,path_name):
    try:
        with open(path_name,'wb') as file:
            # print("save to:",path_name)
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