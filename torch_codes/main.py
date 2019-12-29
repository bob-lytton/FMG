import torch
import torch.nn
import torch.nn.functional as F
from torch import norm
import numpy as np
import time
from model import MF
from loss import MFLoss

gettime = lambda: time.time()

def train_MF(dataloader, reg_user, reg_item, lr=1e-4, epoch=100):
    r"""
    Parameters
    ----------
    dataloader: class Dataloader
        Load metapath adjacent matrix

    epoch: int
        number of epochs
    """
    # set loss function
    loss_func = MFLoss(reg_user, reg_item)

    # load data
    n_user, n_item, adj = dataloader.load()

    # index list
    U = torch.tensor([i for i in range(n_user)]).cuda()
    I = torch.tensor([i for i in range(n_item)]).cuda()

    # initiate forward model
    mf = MF(n_user, n_item).cuda()
    for i in range(epoch):
        adj_t = mf(U, I)
        loss = loss_func(U, I, adj_t, adj)

def train_FM_bpr(dataloader, epoch=50):

    
if __name__ == "__main__":
