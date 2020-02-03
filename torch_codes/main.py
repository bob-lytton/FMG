import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import norm

from loss import MFLoss
from model import FMTrainer, MFTrainer
from utils import *

gettime = lambda: time.time()

def train_MF(metapaths, loadpath, savepath, reg_user=5e-2, reg_item=5e-2, lr=1e-2, epoch=5000, cuda=False):
    r"""
    Parameters
    ----------
    metapaths: list
        list of metapaths

    epoch: int
        number of epochs
    """
    i = 0
    for metapath in metapaths:
        # instance the MF trainer
        trainer = MFTrainer(metapath, loadpath, savepath, epoch[i], cuda=cuda)
        trainer.train(lr=lr[i], reg_user=reg_user[i], reg_item=reg_item[i])
        i += 1

def train_FM(train_X, train_Y, valid_X, valid_Y, epoch):
    trainer = FMTrainer(train_X, train_Y, valid_X, valid_Y)
    trainer.train(epoch)
    
if __name__ == "__main__":
    filtered_path = '../yelp_dataset/filtered/'
    adj_path = '../yelp_dataset/adjs/'
    feat_path = '../yelp_dataset/mf_features/'
    rate_path = '../yelp_dataset/rates'

    # train MF
    metapaths = ['UB', 'UBUB', 'UUB', 'UBCaB', 'UBCiB']
    t0 = gettime()

    train_MF(metapaths, 
             adj_path, 
             feat_path, 
             epoch=[50000, 50000, 50000, 50000, 50000], 
             lr=[2e-4, 5e-4, 1e-4, 1e-4, 1e-4], 
             reg_user=[1e-2, 1e-2, 1e-2, 1e-2, 1e-2], 
             reg_item=[1e-2, 1e-1, 1e-2], cuda=True)

    t1 = gettime()
    print("time cost: %f" % (t1 - t0))

    # train FM (cross entropy loss)

    # first concatenate the embeddings
    # make a user-item matrix of embeddings, size n_user * n_item
    # each cell of the matrix is an embedding of a so called 'sample' in the paper
    # then put the matrix into the model
    # when do we need neg sample?
    users = read_pickle(filtered_path+'users-complete.pickle')
    businesses = read_pickle(filtered_path+'businesses-complete.pickle')
    train_data = read_pickle(rate_path+'train_data.pickle')
    valid_data = read_pickle(rate_path+'valid_data.pickle')
    # valid_neg_data = read_pickle(rate_path+'valid_with_neg_sample.pickle')
    test_data =  read_pickle(rate_path+'test_data.pickle')
    # test_neg_data = read_pickle(rate_path+'test_with_neg_sample.pickle')

    user_features, item_features = load_feature(feat_path, metapaths)
    X = make_embedding(user_features, item_features)    # in this way, we can use X[uid][bid] to find the embedding of user-item pair

    train_Y = train_data
    test_Y = test_data
    valid_Y = valid_data

    train_FM(X, train_Y, X, valid_Y, 1000)