import torch
import torch.nn.functional as F
from torch import norm
import numpy as np
import time
from model import MFTrainer, FMtrainer
from loss import MFLoss
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

def train_FM(dataloader, epoch=50):
    trainer = FMtrainer()
    
if __name__ == "__main__":
    filtered_path = '../yelp_dataset/filtered/'
    matrix_path = '../yelp_dataset/adjs/'
    featurepath = '../yelp_dataset/mf_features/'

    # train MF
    metapaths = ['UB', 'UBUB', 'UUB']
    t0 = gettime()
    train_MF(metapaths, matrix_path, featurepath, epoch=[70000, 70000, 35000], lr=[2e-4, 5e-4, 1e-4], reg_user=[1e-2, 1e-2, 1e-2], reg_item=[1e-2, 1e-1, 1e-2], cuda=True)
    t1 = gettime()
    print("time cost: %f" % (t1 - t0))

    # train FM (cross entropy loss)

    # first concatenate the embeddings
    # make a user-item matrix of embeddings, size n_user * n_item
    # each cell of the matrix is an embedding of a so called 'sample' in the paper
    # then put the matrix into the model
    # finally use the BPR loss as the optimization object
    users = pickle_read(filtered_path+'users.pickle')
    businesses = pickle_read(filtered_path+'businesses.pickle')
    reviews = pickle_read(filtered_path+'reviews.pickle')

    user_features, item_features = load_feature(featurepath, metapaths)
    X = make_embedding(user_features, item_features, users, businesses, reviews)

    train_FM