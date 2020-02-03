import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


def read_pickle(path):
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    return ret

def write_pickle(path, data):
    with open(path, 'wb') as fw:
        pickle.dump(data, fw);

def load_feature(feature_path, metapaths):
    user_features = [read_pickle(feature_path+metapath+'_user.pickle') for metapath in metapaths]
    item_features = [read_pickle(feature_path+metapath+'_item.pickle') for metapath in metapaths]
        
    return user_features, item_features

def make_embedding(user_features, item_features):
    user_concat = torch.cat(user_features, 1)
    item_concat = torch.cat(item_features, 1)
    X = []
    for user in user_concat:
        tmp = []
        for item in item_concat:
            tmp.append(torch.cat([user,item], 0))
        X.append(tmp)

    return X

if __name__ == "__main__":
    pass
