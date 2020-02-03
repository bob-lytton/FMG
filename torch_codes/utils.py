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
        tmp = [torch.cat([user,item], 0).unsqueeze(0) for item in item_concat]
        tmp = torch.cat(tmp, 0)
        X.append(tmp.unsqueeze(0))
    X = torch.cat(X, 0)
    return X

def make_labels(Y, n_user, n_item):
    r"""
    Parameter
    ---------
    Y: list of dict
        saves the interaction information in COO form
    
    Return
    ------
    ret: torch.Tensor
        sparse tensor in COO form
    """
    indices = np.array(([y['user_id'] for y in Y], [y['business_id'] for y in Y]))
    values = np.array([1. for y in Y])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ret = torch.sparse_coo_tensor(indices, values, size=(n_user,n_item),
                                  dtype=torch.float32, device=device, requires_grad=False)
    return ret

if __name__ == "__main__":
    pass
