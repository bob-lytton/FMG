import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

def pickle_read(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def load_feature(feature_path, metapaths):
    user_features = []
    item_features = []
    for metapath in metapaths:
        user_file = feature_path + metapath + '_user.pickle'
        item_file = feature_path + metapath + '_item.pickle'
        with open(user_file, 'rb') as f:
            user_features.append(pickle.load(f))
        with open(item_file, 'rb') as f:
            item_features.append(pickle.load(f))
        
    return user_features, item_features

def make_embedding(user_features, item_features):
    user_concat = torch.cat(user_features, 1)
    item_concat = torch.cat(item_features, 1)
    X = []
    for user in user_concat:
        for item in item_concat:
            X.append(torch.cat([user,item], 1))

    return X

if __name__ == "__main__":
    pass