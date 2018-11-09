#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:41 2018

@author: thalita

ldmnet with skorch callback

"""

import numpy as np

from scipy import sparse
from skorch_utils import NNClassifier, StopperNet
from skorch.utils import to_tensor
from skorch.callbacks import Callback
import torch

from laplacian_utils import compute_W, compute_L


class LDMnetBase(StopperNet, NNClassifier):
    def __init__(self,
                 module,
                 layer_name,
                 mu=0.1,
                 epochs_update=2,
                 lambda_bar=0.01,
                 criterion=torch.nn.CrossEntropyLoss,
                 **kwargs):
        super().__init__(module, criterion=criterion, **kwargs)
        self.layer_name = layer_name
        self.epochs_update = epochs_update
        self.mu = mu
        self.lambda_bar = lambda_bar

    def initialize(self):
        super().initialize()
        self.ksi = None
        self.Z = None
        self.alpha = None
        self.train_ = None


    def _init_ksi_Z_alpha(self, X):
        self.ksi = self.transform(X)
        self.Z = np.zeros_like(self.ksi)
        self.alpha = np.zeros_like(self.ksi)

    def transform(self, X):
        return super().transform(X, name=self.layer_name)

    def fit(self, X, y, X_imgs=None, **fit_args):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        self._init_ksi_Z_alpha(X)
        input_dict = dict(X=X,
                          ksi=self.ksi,
                          Z=self.Z,
                          alpha=self.alpha)
        if X_imgs:
            input_dict.update(X_imgs=X_imgs)

        while len(self.history) < self.max_epochs and not self.stop:
            super().partial_fit(input_dict,
                                y, epochs=self.epochs_update,
                                **fit_args)

        return self


class LDMnet(LDMnetBase):
    def initialize_callbacks(self):
        self.callbacks += [AlphaUpdate(layer_name=self.layer_name,
                                       lambda_bar=self.lambda_bar,
                                       epochs_update=self.epochs_update,
                                       mu=self.mu)]
        super().initialize_callbacks()

    def get_loss(self, y_pred, y_true, X=None, training=False):
        if isinstance(X, dict):
            X_X = X['X']
        else:
            X_X = X
        loss = super().get_loss(y_pred, y_true, X_X, training)
        if not training:
            return loss
        if X is not None and self.alpha is not None and self.lambda_bar != 0:
            ksi = self.infer(X, name=self.layer_name)
            alpha = to_tensor(X['alpha'], device=self.device).to(X_X.dtype)
            Z = to_tensor(X['Z'], device=self.device).to(X_X.dtype)
            reg_loss = self.mu * 0.5 * torch.norm(alpha - ksi + Z)
            reg_loss = reg_loss.mean(dim=0)
            loss += reg_loss
        return loss


class AlphaUpdateBase(object):
    def _solve_lin_sys(self, W, L, ksij, Zj):
        """solve linear system for alpha_j

        (L + c W) alpha_j = c W (ksi_j - Z_j) (eq 19)

        where c = mu/lambda_bar
        """
        c = self.mu/self.lambda_bar
        v = (ksij-Zj)
        # Solve Ax = b
        cW = c*W
        A = (L + cW).tocsc()
        b = cW * v
        # start alpha_j as ksi_j
        x0 = ksij
        # preconditioning matrix M should approximate inv A
        # spilu returns SuperLU object with solve(b) that approx solves Ax=b
        # To improve the better approximation to the inverse, you may need to
        # increase `fill_factor` AND decrease `drop_tol`.
        if self.preconditionner:

            M_approx = sparse.linalg.spilu(A, drop_tol=1e-4, fill_factor=10)
            M = sparse.linalg.LinearOperator(
                shape=A.shape, matvec=M_approx.solve)
        else:
            M = None
        x, info = sparse.linalg.cg(A, b, x0, M=M,
                                   tol=self.tol, maxiter=self.max_iter)
        self.solver_info_.append(info)
        return x

    def _cat_ksi(self, ksi,  X, X_imgs=None):
        '''
        X : samples (images or features)
        X_imgs : input images (in case X contains pre-extracted features)
        '''
        if self.concatenate_input:
            input = X if X_imgs is None else X_imgs
            return np.concatenate([ksi, input.view(input.shape[0],-1)],
                                   axis=-1)
        else:
            return ksi

    def _update_W_L(self, ksi, X, X_imgs=None):
        cat_ksi = self._cat_ksi(ksi, X, X_imgs=X_imgs)
        self.W_ = compute_W(cat_ksi, self.n_neighbors,
                            nn_radius=10)
        self.L_ = compute_L(self.W_)

    def _update_alpha(self, net):
        n_features = net.ksi.shape[1]

        for j in range(n_features):
            alphaj = self._solve_lin_sys(self.W_, self.L_,
                                         net.ksi[net.train_, j],
                                         net.Z[net.train_, j])
            net.alpha[net.train_, j] = alphaj


class AlphaUpdate(Callback, AlphaUpdateBase):
    def __init__(self, mu=0.01, lambda_bar=0.01, n_neighbors=20,
                 tol=1e-5, max_iter=50, n_jobs=1,
                 epochs_update=2,
                 preconditionner=False,
                 concatenate_input=True,
                 *args, **kwargs):
        '''
        - mu: multiplier for the alternating direction method of multipliers (ADMM)
        - l or lambda_: regularization + temperature
        $\hat{\lambda} = t/2\lambda = (8 \lambda\gamma)^{-1}$
            - lambda: regularization strenth
            - t or gamma: heat kernel param, $\gamma=\frac{1}{4t}$
        - n_neighbors: for kNN graph
        - max_iter: max iterations for lin sys solver
        - tol: tolerance for lin sys solver
        - n_jobs: num of jobs for nn graph construction
        '''
        self.concatenate_input = concatenate_input
        self.epochs_update = epochs_update
        self.mu = mu
        self.lambda_bar = lambda_bar
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.preconditionner = preconditionner

    def initialize(self):
        super().initialize()
        self.solver_info_ = []
        self.W_ = None # W and L have size of training set
        self.L_ = None

    def on_train_begin(self, net, X, y=None, **kargs):
        if len(net.history) == 1:
            net._init_ksi_Z_alpha(X)
        # if 'X_imgs' in X:
        #     X_imgs = X['X_imgs']
        # else:
        #     X_imgs = None
        # X = X['X']
        # if self.lambda_bar != 0:
        #     # compute W, L and solve linsys to update alpha
        #     self._update_W_L(net.ksi, X, X_imgs)
        #     self._update_alpha(net)

    def on_epoch_begin(self, net,
                       dataset_train, dataset_valid=None, **kwargs):
        epochs = len(net.history)
        if not (epochs-1) % self.epochs_update:
            net.train_ = dataset_train.indices
            X, y = dataset_train[:]
            if 'X_imgs' in X:
                X_imgs = X['X_imgs']
            else:
                X_imgs = None
            X = X['X']
            if self.lambda_bar != 0:
                # compute W, L and solve linsys to update alpha
                self._update_W_L(net.ksi[net.train_], X, X_imgs)
                self._update_alpha(net)

    def on_epoch_end(self, net,
                     dataset_train, dataset_valid=None, **kwargs):
        epochs = len(net.history)
        if not (epochs-1) % self.epochs_update:
            X, y = dataset_train[:]
            net.ksi[net.train_] = net.transform(X['X'])

            # dual variable update
            dZ = net.alpha[net.train_] - net.ksi[net.train_]
            net.Z[net.train_] = net.Z[net.train_] + dZ

    def on_train_end(self, net, X, y=None, **kwargs):
        pass
        # net.ksi[...] = net.transform(X)

        # # dual variable update
        # dZ = net.alpha - net.ksi
        # net.Z[...] = net.Z + dZ

class SaveVars(Callback):
    """ Callback to save ksi, alpha and Z"""

    def __init__(self, every_n_epochs=100):
        self.every_n_epochs = every_n_epochs

    def initialize(self):
        self.ksi = []
        self.Z = []
        self.alpha = []
        self.epochs = []
        return self

    def on_train_end(self, net, **kwargs):
        """
        Method notified after partial fit, when it is interesting to save
        ldmnet vars.
        """
        epochs = len(net.history)
        if not (epochs-1) % self.every_n_epochs:
            self.ksi.append(net.ksi.copy())
            self.Z.append(net.Z.copy())
            self.alpha.append(net.alpha.copy())
            self.epochs.append(epochs - 1)
