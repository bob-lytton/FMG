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

def make_omega(origin_data, n_users, n_items, mode):
    r"""
    Parameters
    ----------
    origin_data: list of dict

    mode: 'train' or 'valid'

    Return
    ------
    omega: np.ndarray
    """
    omega = np.zeros([n_users, n_items])
    if mode == 'train':
        for d in origin_data:
            omega[d['user_id']][d['business_id']] = d['rate']
    
    elif mode == 'valid':
        for d in origin_data:
            for bid in d['pos_business_id']:
                omega[d['user_id']][bid] = 1.0

if __name__ == "__main__":
    pass
