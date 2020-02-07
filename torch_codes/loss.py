import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, norm


class MFLoss(nn.Module):
    def __init__(self, reg_user, reg_item):
        super(MFLoss, self).__init__()
        self.reg_user = reg_user / 2.0
        self.reg_item = reg_item / 2.0

    def forward(self, user_mat, item_mat, adj_predicted, adj):
        r"""
        Parameters
        ----------
        user_mat: torch.Tensor

        item_mat: torch.Tensor

        adj_predicted: torch.Tensor

        adj: torch.Tensor
        """
        return 0.5 * norm(adj_predicted - adj, p='fro') + \
            self.reg_user * norm(user_mat, p='fro') + \
            self.reg_item * norm(item_mat, p='fro')
