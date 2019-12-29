import os
import pickle
import time

import numpy as np
import torch
from torch import nn

gettime = lambda: time.time()

class MF(nn.Module):
    def __init__(self, n_user, n_item, K=10, iters=500, cuda=False):
        r"""
        Parameters
        ----------
        n_user: int
            number of users

        n_item: int
            number of items

        K: int
            embedding dim

        iters: int
            number of iteration
        """
        super(MF, self).__init__()
        self.iters = iters
        self.n_user = n_user
        self.n_item = n_item
        self.n_factor = K
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.user_factors = torch.nn.Embedding(n_user, K, sparse=True)
        self.item_factors = torch.nn.Embedding(n_item, K, sparse=True)

    def forward(self, user, item):
        r"""
        Parameters
        ----------
        user: int
            index to fine embedding of the corresponding factors

        item: int
            index to fine embedding of the corresponding factors

        Return
        ------
        adj_t: Tensor with shape n_user*n_item, 
            the predicted adjacent matrix
        """

        # to calculate matrix factorization, we need
        # to compute the adj matrix by multiplying
        # the two embedding matrices

        U = self.user_factors(user)
        I = self.item_factors(item)
        return U.mm(I.T)

    def export(self, user_file, item_file):
        r"""
        export the embeddings to files
        """
        with open(user_file+'.pickle', 'wb') as fw:
            pickle.dump(self.user_factors, fw)
        with open(item_file+'.pickle', 'wb') as fw:
            pickle.dump(self.item_factors, fw)


if __name__ == "__main__":
    # Test function
    n_user = 1000
    n_item = 1000
    K = 10
    U = torch.tensor([i for i in range(n_user)]).cuda()
    I = torch.tensor([i for i in range(n_item)]).cuda()
    mf = MF(n_user, n_item).cuda()

    t0 = gettime()
    D = mf(U, I)
    t1 = gettime()
    print("time cost:", t1 - t0)
    print(D)
