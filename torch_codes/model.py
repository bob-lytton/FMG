import copy
import os
import pickle
import random
import time

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from loss import MFLoss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from utils import read_pickle, write_pickle

print(os.listdir("../input"))


gettime = lambda: time.time()

class MatrixFactorizer(nn.Module):
    """
    Updating Embeddings within the model is OK
    """
    def __init__(self, n_user, n_item, n_factor=10, cuda=False):
        r"""
        Parameters
        ----------
        n_user: int
            number of users

        n_item: int
            number of items

        n_factor: int
            embedding dim
        """
        super(MatrixFactorizer, self).__init__()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.user_factors = torch.randn([n_user, n_factor], dtype=torch.float32, requires_grad=True, device=self.device)
        self.item_factors = torch.randn([n_item, n_factor], dtype=torch.float32, requires_grad=True, device=self.device)

    def forward(self):
        r"""
        Return
        ------
        adj_t: Tensor with shape n_user*n_item, 
            the predicted adjacent matrix
        """
        # to calculate matrix factorization, we need
        # to compute the adj matrix by multiplying
        # the two embedding matrices
        return self.user_factors.mm(self.item_factors.T)

    def export(self, filepath, metapath):
        r"""
        export the embeddings to files
        """
        user_file = filepath + metapath + '_user' + '.pickle'
        item_file = filepath + metapath + '_item' + '.pickle'
        write_pickle(user_file, self.user_factors)
        write_pickle(item_file, self.item_factors)

class MFTrainer(object):
    r"""
    Training wrapper of MatrixFactorizer
    """
    def __init__(self, metapath, loadpath, savepath, epochs=20, n_factor=10, is_binary=True, cuda=False):
        self.metapath = metapath
        self.loadpath = loadpath
        self.savepath = savepath
        self.epochs = epochs
        self.n_factor = n_factor
        self.cuda = cuda
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.n_user, self.n_item, self.adj_mat = self._load_data(loadpath, metapath, is_binary)

        # instance model
        self.mf = MatrixFactorizer(self.n_user, self.n_item, self.n_factor, self.cuda)
    
    def _load_data(self, filepath, metapath, is_binary=True):
        r"""
        Parameters
        ----------
        filepath: str

        metapath: str

        is_binary: bool, 
            if the files are binary files
        
        Return
        ------
        n_user: int
            number of users

        n_item: int
            number of businesses

        data: torch.Tensor(requires_grad=False)
            the adjacency matrix to be factorized
        """
        data = []
        if is_binary == True:
            file = filepath + 'adj_' + metapath
            with open(file, 'rb') as fw:
                adjacency = pickle.load(fw)
                data = torch.tensor(adjacency, dtype=torch.float32, requires_grad=False).to(self.device)
        if is_binary == False:
            """ TODO: read txt file """
            raise NotImplementedError

        n_user, n_item = data.shape
        return n_user, n_item, data

    def _export(self, filepath, metapath):
        r"""
        export the matrix factors to files
        """
        self.mf.export(filepath, metapath)

    def train(self, lr=1e-4, reg_user=5e-2, reg_item=5e-2, decay_step=30, decay=0.1):
        r"""
        Parameters
        ----------
        lr: learning rate

        reg_user: regularization coefficient

        reg_item: regularization coefficient

        TODO:
        -----
        add lr scheduler
        """
        # set loss function
        criterion = MFLoss(reg_user, reg_item).to(self.device)
        optimizer = torch.optim.Adam([self.mf.user_factors, self.mf.item_factors], lr=lr)  # use weight_decay
        # scheduler = StepLR(optimizer, step_size=decay_step, gamma=decay)
        self.mf.train()
        print("n_user: %d, n_item: %d" % (self.n_user, self.n_item))
        for i in range(self.epochs + 1):
            # scheduler.step()
            self.mf.zero_grad()
            adj_t = self.mf()
            loss = criterion(self.mf.user_factors, self.mf.item_factors, adj_t, self.adj_mat)   # this line is ugly
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("metapath: %s, epoch %d: loss = %.4f, lr = %.10f, reg_user = %f, reg_item = %f" 
                    % (self.metapath, i, loss, lr, reg_user, reg_item))
                
        self._export(self.savepath, self.metapath)


class FactorizationMachine(nn.Module):
    def __init__(self, n=None, k=None):
        r"""
        Parameters
        ----------
        n: int
            number of embeddings, n = n_user + n_item
        
        k: int
            dimension of each embedding
        """
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)  # nn.Linear also contains the bias

    def forward(self, x):
        r"""
        Parameters
        ----------
        x: torch.Tensor.sparse, shape 1*((n_user+n_item)*K)
            input embedding of each user-business pair, saved in COO form
        
        Return
        ------
        out: scalar
            prediction of y
        """
        out_1 = torch.matmul(x, self.V).pow(2).sum(1, keepdim=True) # S_1^2, S_1 can refer to statistics book
        out_2 = torch.matmul(x.pow(2), self.V.pow(2)).sum(1, keepdim=True) # S_2
        
        out_inter = 0.5*(out_1 - out_2) # sum(<vi, vj>*xi*xj)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out

    def export(self):
        path = 'yelp_dataset/fm_res/'
        V_filename = 'FM_V.pickle'
        lin_filename = 'FM_lin.pickle'
        write_pickle(path+V_filename, self.V)
        write_pickle(path+lin_filename, self.lin)

    def load(self, filenames):
        r"""
        load parameters from files
        """
        V_file, lin_file = filenames
        self.V = read_pickle(V_file)
        self.lin = read_pickle(lin_file)


class FMTrainer(object):
    def __init__(self, train_X, train_Y, valid_X, valid_Y, criterion=None):
        r"""
        Parameters
        ----------
        model: nn.Module
            the FM model

        train_X: torch.tensor
            each line of train_X is the concatenation of the embeddings of each user and item,
            and represents a user-item pair

        criterion: loss function class,
            Default is nn.CrossEntropyLoss
        """
        self.train_X = train_X
        self.train_Y = train_Y
        self.valid_X = valid_X
        self.valid_Y = valid_Y
        self.FM = FactorizationMachine(train_X.shape[0], 10)
        self.criterion = nn.CrossEntropyLoss()
        if criterion is not None:
            self.criterion = criterion

    def _export(self):
        self.FM.export()

    def train(self, nepoch):
        # set optimizer
        optimizer = torch.optim.Adam([self.FM.V, self.FM.lin], lr=lr)  # use weight_decay
        for epoch in range(nepoch):
            for x, y in self.train_X, self.train_Y:
                y_t = self.FM(x)
                loss = self.criterion(y_t, y)
                loss.backward()
                optimizer.step()

            if epoch % 50 == 0:
                print("epoch %d, loss = %f, lr = %f" % (epoch, loss, lr))
            if epoch % 100 == 0:
                # valid evaluate

        self._export()


if __name__ == "__main__":
    # Test function
    pass