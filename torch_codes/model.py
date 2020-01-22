import os
import pickle
import time

import numpy as np  # linear algebra
import torch
from torch import nn
from loss import MFLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import copy
print(os.listdir("../input"))


gettime = lambda: time.time()

class MatrixFactorizer(nn.Module):
    """
    Updating Embeddings within the model is OK
    """
    def __init__(self, n_user, n_item, n_factor=10, iters=500, cuda=False):
        r"""
        Parameters
        ----------
        n_user: int
            number of users

        n_item: int
            number of items

       n_factor: int
            embedding dim

        iters: int
            number of iteration
        """
        super(MatrixFactorizer, self).__init__()
        self.iters = iters
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = n_factor
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
        user_file = filepath + metapath + '_user'
        item_file = filepath + metapath + '_item'
        with open(user_file+'.pickle', 'wb') as fw:
            pickle.dump(self.user_factors, fw)
        with open(item_file+'.pickle', 'wb') as fw:
            pickle.dump(self.item_factors, fw)

class MFTrainer(object):
    r"""
    Training wrapper of MatrixFactorizer
    """
    def __init__(self, metapath, loadpath, savepath, epoch=20, n_factor=10, iters=500, is_binary=True, cuda=False):
        self.metapath = metapath
        self.loadpath = loadpath
        self.savepath = savepath
        self.epoch = epoch
        self.n_factor = n_factor
        self.iters = iters
        self.cuda = cuda
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.n_user, self.n_item, self.adj_mat = self._load_data(loadpath, metapath, is_binary)

        # instance model
        self.mf = MatrixFactorizer(self.n_user, self.n_item, self.n_factor, self.iters, self.cuda)
    
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
        for i in range(self.epoch + 1):
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
    r"""
    Parameters
    ----------
    n: int
        number of embeddings, n = n_user + n_item
    
    k: int
        dimension of each embedding
    """
    def __init__(self, n=None, k=None):
        super().__init__()
        # Initially we fill V with random values sampled from Gaussian distribution
        # NB: use nn.Parameter to compute gradients
        self.V = nn.Parameter(torch.randn(n, k),requires_grad=True)
        self.lin = nn.Linear(n, 1)

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
        
        out_inter = 0.5*(out_1 - out_2)
        out_lin = self.lin(x)
        out = out_inter + out_lin
        
        return out

class FMTrainer(object):
    def __init__(self, model, train_X, train_Y, test_X, test_Y):
        r"""
        Parameters
        ----------
        model: nn.Module
            the FM model

        train_X: torch.tensor
            each line of train_X is the concatenation of the embeddings of each user and item,
            and represents a user-item pair
        """
        self.FM = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

    def train(self, nepoch):
        for epoch in range(nepoch):
            self.FM()

class BayesianPersonalizedRanking(nn.Module):
    	def __init__(self, user_num, item_num, factor_num):
    		super(BPR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item_i, item_j):
		user = self.embed_user(user)
		item_i = self.embed_item(item_i)
		item_j = self.embed_item(item_j)

		prediction_i = (user * item_i).sum(dim=-1)
		prediction_j = (user * item_j).sum(dim=-1)
		return prediction_i, prediction_j
    
class BPRTrainer(object):
    def __init__(self, model, train_X, train_Y, test_X, test_Y):
        self.BPR = model
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
    
    def train(self, nepoch):
        for epoch in range(nepoch):
            pass

if __name__ == "__main__":
    # Test function
    n_user = 1000
    n_item = 1000
    n_factor = 10
    U = torch.tensor([i for i in range(n_user)]).cuda()
    I = torch.tensor([i for i in range(n_item)]).cuda()
    mf = MatrixFactorizer(n_user, n_item).cuda()
    t0 = gettime()
    D = mf(U, I)
    t1 = gettime()
    print("time cost:", t1 - t0)
    print(D)
