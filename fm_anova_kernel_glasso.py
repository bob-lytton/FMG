#coding=utf8
'''
    standard fm, i.e. poly regression with anova kernel
    regularization is group lasso
'''
import time
import logging

import numpy as np
from numpy.linalg import norm

from exp_util import cal_rmse, cal_mae

stf = lambda eta, nw: 1.0 - eta / nw if eta < nw else 0.0 # soft threshold function

class FMAKGL(object):
    # TODO: change loss, modify update_param and cal_loss
    # need to sample a negative sample I_j

    def __init__(self, config, data_loader, seed=1234):
        self.config = config
        self.seed = seed
        self.train_X, self.train_Y, self.test_X, self.test_Y, self.item_num = data_loader.get_exp_data()
        self._init_config()

    def _init_config(self):
        self.exp_id = self.config.get('exp_id')
        self.N = self.config.get('N')
        self.K = self.config.get('K')
        self.L = self.config.get('L')
        self.F = self.config.get('F')
        self.initial = self.config.get('initial')
        self.reg_W = self.config.get('reg_W')
        self.reg_P = self.config.get('reg_P')
        self.max_iters = self.config.get('max_iters')
        self.ln = self.config.get('ln')
        self.eps = self.config.get('eps')
        self.eta = self.config.get('eta')
        self.solver = self.config.get('solver')
        self.bias_eta = self.config.get('eta')
        self.bias = np.mean(self.train_Y)
        #better to add log information for the configs

        self.M = self.train_X.shape[0]

    def _prox_op(self, eta, G, g_inds):
        for i in range(len(g_inds)):
            G[g_inds[i]] = stf(eta, norm(G[g_inds[i]])) * G[g_inds[i]]
        return G

    def _group_lasso(self, G, g_inds):
        res = 0.0
        for i in range(g_inds.shape[0]):
            res += norm(G[g_inds[i]])
        return res

    def _obj(self, err, W, P):
        part1 = np.power(err, 2).sum() / self.M
        part2 = self.reg_W * self._group_lasso(W, self.gw_inds)
        part3 = self.reg_P * self._group_lasso(P.flatten(), self.gp_inds)
        logging.debug('obj detail, part1=%s, part2=%s, part3=%s', part1, part2, part3)
        return part1 + part2 + part3

    def _cal_err(self, WX, XP, XSPS, Y):
        Y_t = self.bias + WX + 0.5 * (np.square(XP) - XSPS).sum(axis=1)
        return Y_t - Y

    def _get_XC_prods(self, X, W, P):
        WX = np.dot(W, X.T)
        XP = np.dot(X, P)
        XSPS = np.dot(np.square(X), np.square(P))
        return WX, XP, XSPS

    def get_eval_res(self):
        return self.rmses, self.maes

    def train(self):
        W = np.random.rand(len(self.train_X), self.K) * self.initial
        H = np.random.rand(self.item_num, self.K) * self.initial   # N by K
        self._fm_with_bpr(W, H)

    def _get_updated_paras(self, eta, W, P):
        # TODO: modify gradient
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        obj_t = self._obj(err, W, P)

        #cal gradients
        grad_W = 2.0 / self.M * np.dot(err, self.train_X)#element-wise correspondence

        XS = np.square(self.train_X)
        grad_P = np.zeros(P.shape)
        for f in range(self.K):
            grad_P[:,f] = 2.0 / self.M * np.dot(err, np.multiply(self.train_X, XP[:,f].reshape(-1,1).repeat(self.N, axis=1)) - np.multiply(P[:,f].reshape(1, -1).repeat(self.M, axis=0), XS))

        l_obj, eta, lt, W, P = self._line_search(obj_t, eta, W, P, grad_W, grad_P)

        return l_obj, eta, lt, W, P

    def _update_bias(self, W, P):
        WX, XP, XSPS = self._get_XC_prods(self.train_X, W, P)
        err = self._cal_err(WX, XP, XSPS, self.train_Y)
        self.bias -= self.bias_eta * 2.0 / self.M * err.sum()

    def _line_search(self, obj_v, eta, W, P, grad_W, grad_P):
        for lt in range(self.ln+1):
            lW = W - eta * grad_W
            lW = self._prox_op(eta * self.reg_W, lW, self.gw_inds)

            lP = P - eta * grad_P
            lP = self._prox_op(eta * self.reg_P, lP.flatten(), self.gp_inds)
            lP = lP.reshape(P.shape)

            lWX, XlP, XSlPS = self._get_XC_prods(self.train_X, lW, lP)
            l_err = self._cal_err(lWX, XlP, XSlPS, self.train_Y)
            l_obj = self._obj(l_err, lW, lP)

            if l_obj < obj_v:
                eta *= 1.1
                W, P = lW, lP
                break
            else:
                eta *= 0.7
        return l_obj, eta, lt, W, P

    def _save_paras(self, W, P):
        split_num = self.config['sn']
        dt = self.config.get('dt')
        W_wfilename = 'fm_res/%s_split%s_W_%s_exp%s.txt' % (dt, split_num, self.reg_W, self.exp_id)
        np.savetxt(W_wfilename, W)
        P_wfilename = 'fm_res/%s_split%s_P_%s_exp%s.txt' % (dt, split_num, self.reg_P, self.exp_id)
        np.savetxt(P_wfilename, P)
        logging.info('W and P saved in %s and %s', W_wfilename, P_wfilename)

    def _fm_with_bpr(self, W, H):
        """
        Parameter
        ---------
        W: N*k matrix
            matrix of user embedding
        H: N*k matrix
            matrix of item embedding
        """
        indptr = [u for u in range(self.item_num)]

        sampled_pos, sampled_neg = self._sample(len(self.train_X), self.item_num, )
        
    def _sample(self, n_users, n_items, indices, indptr):
        """
        sample batches of random triplets (u, i, j)
        """
        sampled_pos_items = np.zeros(self.bsz, dtype=np.int)
        sampled_neg_items = np.zeros(self.bsz, dtype=np.int)
        sampled_users = np.random.choice(n_users, size=self.bsz, replace=False)

        for idx, user in enumerate(sampled_users):
            pos_items = indices[indptr[user]: indptr[user+1]]
            pos_item = np.random.choice(pos_items)  # single value
            neg_item = np.random.choice(n_items)    # single value
            while neg_item in pos_items:
                neg_item = np.random.choice(n_items)

            sampled_pos_items[idx] = pos_item
            sampled_neg_items[idx] = neg_item
        return sampled_users, sampled_pos_items, sampled_neg_items
