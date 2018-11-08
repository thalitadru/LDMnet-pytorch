#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:46:32 2018

@author: thalita
"""

from sklearn.base import TransformerMixin, ClassifierMixin
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from skorch.utils import duplicate_items
from skorch.utils import get_dim
from skorch.utils import to_tensor, to_numpy
from skorch.utils import params_for
from skorch.callbacks import Callback
import skorch.callbacks.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from skorch.net import NeuralNet
from skorch.classifier import NeuralNetClassifier
from functools import partial
from tempfile import mktemp
from collections import defaultdict

import matplotlib.pyplot as plt


class StopperNet(object):
    """
    StopperNet.

    This class captures the keyboardInterrupt exception during the training
    loop, setting the attribute stop to true.
    Any model that needs to run fit() by mutiple calls to partial_fit() can use
    this class to know when this exception has been raised and stop training.
    Normal skorch behavior is to capture the exception and do nothing.
    """

    def initialize(self):
        super().initialize()
        self.stop = False

    def partial_fit(self, X, y=None, classes=None, **fit_params):
        """Fit the module.

        If the module is initialized, it is not re-initialized, which
        means that this method should be used if you want to continue
        training a model (warm start).

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * a dictionary of the former three
            * a list/tuple of the former three

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported.

        classes : array, sahpe (n_classes,)
          Solely for sklearn compatibility, currently unused.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the train_split call.

        """
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y, **fit_params)
        except KeyboardInterrupt:
            self.stop = True
        self.notify('on_train_end', X=X, y=y)
        return self


class TransformerNet(TransformerMixin):
    def initialize(self):
        if not self.initialized_:
            super().initialize()
        self.transform_args = None

    def transform(self, X, **forward_kwargs):
        self.transform_args = forward_kwargs
        out = self.predict_proba(X)
        self.transform_args = None
        return out

    def evaluation_step(self, Xi, training=False):
        """Perform a forward step to produce the output used for
        prediction and scoring.

        Therefore the module is set to evaluation mode by default
        beforehand which can be overridden to re-enable features
        like dropout by setting ``training=True``.

        """
        self.module_.train(training)
        if self.transform_args is not None:
            return self.infer(Xi, **self.transform_args)
        else:
            return self.infer(Xi)

class MLP(nn.Module):
    def __init__(self, n_in=2, num_units=20, n_out=10,
                 drop_proba=0.5, nonlin=F.relu):
        super().__init__()
        if num_units != 0:
            if type(num_units) is int:
                num_units = tuple([num_units,])
            self.n_hidden = len(num_units)
            self.__setattr__('hidden1', nn.Linear(n_in, num_units[0]))
            prev_layer = num_units[0]
            for layer_ix in range(1, len(num_units)):
                this_layer = num_units[layer_ix]
                self.__setattr__('hidden%d' % (layer_ix+1),
                                      nn.Linear(prev_layer, this_layer))
                prev_layer = this_layer
        else:
            self.n_hidden = 0
            num_units = [n_in]

        self.nonlin = nonlin
        if drop_proba is not None and drop_proba != 0:
            self.dropout = nn.Dropout(drop_proba)
        else:
            self.dropout = None
        self.output = nn.Linear(num_units[-1], n_out)


    def forward(self, X, name='output', **kwargs):
        if self.n_hidden >= 1:
            for i in range(self.n_hidden):
                layer_name = 'hidden%d' % (i + 1)
                layer = self.__getattr__(layer_name)
                X = layer(X)
                X = self.nonlin(X)
                if name == layer_name:
                    return X
        if self.dropout is not None:
            X = self.dropout(X)
        if name != 'output':
            raise Warning("name %s dos not correspont to any layers," % name +
                          " returning output" )
        #X = F.log_softmax(self.output(X), dim=-1)
        return X


class NNClassifier(TransformerNet, NeuralNetClassifier, ClassifierMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def learning_curve(model):
    net = model
    loss = [net.history['train_loss', i] for i in range(len(net.history))]
    valid_loss = [net.history['valid_loss', i] for i in range(len(net.history))]
    val_acc = [net.history['valid_acc', i] for i in range(len(net.history))]
    plt.figure()
    plt.plot(loss, linestyle=':', label='train')
    lines = plt.plot(valid_loss, linestyle=':', label="valid")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc='best')
    ax2 = plt.gca().twinx()
    ax2.plot(val_acc, label='valid', color=lines[0].get_color())
    plt.ylabel('accuracy')
    plt.title('Learning curve')


class SaveWeights(Callback):
    def __init__(self, every_n_epochs=100):
        self.every_n_epochs = every_n_epochs

    def initialize(self):
        self.params = []
        self.epochs = []
        return self

    def on_epoch_end(self, net, **kwargs):
        epochs = len(net.history)
        if not (epochs % self.every_n_epochs):
            p_t = list(net.module_.named_parameters())
            p_t = dict(p_t)
            for k, v in p_t.items():
                p_t[k] = to_numpy(v)
            self.params.append(p_t)
            self.epochs.append(epochs - 1)

class GradientInspector(Callback):
    def __init__(self, frequency=100):
        self.frequency = frequency

    def initialize(self):
        super().initialize()
        self.grads = defaultdict(list)

    def on_grad_computed(self, net, named_parameters, **kwargs):
        epochs = len(net.history)
        if not(epochs % self.frequency):
            for name, par in named_parameters:
                self.grads[name].append(to_numpy(par.grad))


class NaNStopping(Callback):
    def on_epoch_end(self, net, **kwargs):
        if np.isnan(net.history[-1, 'train_loss']):
            print("NaN stopping @ ", len(net.history))
            raise KeyboardInterrupt
