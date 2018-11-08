#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 11:07:10 2018

@author: thalita
"""

import os
import numpy as np
import torch
import torchvision
from torchvision.datasets import MNIST, SVHN, CIFAR10
from sklearn.model_selection import StratifiedShuffleSplit
import sacred
import skorch
from skorch.callbacks import LRScheduler
from cnn_models import CIFAR10net, MNISTnet
from LDMnet import LDMnet

DATA_DIR = './Data'
#%%
def flatten_dict(d, preffix=''):
    new_d = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if preffix:
                pref = preffix + '__' + k
            else:
                pref = k
            new_d.update(flatten_dict(v, pref))
    else:
        new_d.update({preffix: d})
    return new_d

# %% Experiment
name = 'ldmnet_runs'
ex = sacred.Experiment(name)
observer = sacred.observers.FileStorageObserver.create(name)
ex.observers.append(observer)
# %% Configs
@ex.config
def base_config():
    # pylint: disable=unused-variable
    seed = 15646842
    batch_size = 100
    max_epochs = 500
    weight_decay = 0.0
    dataset = 'mnist'
    layer_name = 'conv3'
    lr = 0.001
    mu = 0.001
    lambda_bar = 0.01
    train_size = 500
    device = 'cpu' # cpu or cuda
    dropout = 0.0
    epochs_update=2
    optimizer = dict(weight_decay=weight_decay, momentum=0.9)
    alphaupdate = dict(
        n_neighbors=20,
        tol=1e-5,
        max_iter=50,
        preconditionner=None,
        n_jobs=2,  # num jobs for nn graph construction
        concatenate_input=True)
    module = dict(dropout=dropout)

@ex.named_config
def mnist():
    train_size = 500


    dataset = 'mnist'

    layer_name='conv3'

    mu = 0.001
    lr = 0.001
    lambda_bar = (0.05 if train_size < 400 else
                  (0.01 if train_size < 1000 else
                   (0.005 if train_size < 3000 else 0.001)))


    weight_decay =  (0.1 if train_size < 100 else
                     (0.05 if train_size < 400 else
                      (0.01 if train_size < 700 else
                       (0.005 if train_size < 3000 else 0.001))))


@ex.named_config
def cifar10():
    train_size = 500

    dataset = 'cifar10'

    layer_name='fc2'

    lr = 0.001
    mu = 1.0
    lambda_bar = 0.01

    weight_decay = (5e-4 if train_size < 100 else
                    (5e-5 if train_size < 700 else 5e-7))

@ex.named_config
def svhn():
    train_size = 500

    dataset = 'svhn'

    layer_name='fc2'

    lr = 0.005
    mu = 0.5
    lambda_bar = (0.1 if train_size < 100 else
                  (0.05 if train_size < 700 else 0.01))

    weight_decay = (1e-6 if train_size < 400 else
                    (1e-7 if train_size < 700 else 1e-8))

# %% Dataset
@ex.capture
def get_dataset(dataset, train_size, _log, training=True):
    _log.info("Loading dataset...")
    if dataset.lower() == 'mnist':
        Dataset = MNIST
    elif dataset.lower() == 'cifar10':
        Dataset = CIFAR10
    elif dataset.lower() == 'svhn':
        Dataset = SVHN
    else:
        raise ValueError("invalid dataset name: %s" % dataset)

    path = os.path.join(DATA_DIR, dataset)
    ds = Dataset(path, train=training, download=True,
                 transform=torchvision.transforms.ToTensor())

    if training:
        splitter = StratifiedShuffleSplit(n_splits=1,
                                          train_size=train_size)
        dummyX = np.zeros([len(ds), 1])
        y = np.array(ds.train_labels)
        indices, _ = next(splitter.split(dummyX, y))
        X = torch.stack([ds[i][0] for i in indices])
        labels = y[indices]
    else:
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        labels = np.array(ds.test_labels)
    _log.info("Loading dataset: Done!")
    return X, labels


@ex.capture
def get_module(dataset):
    if dataset.lower() == 'mnist':
        module = MNISTnet
    elif dataset.lower() == 'cifar10' or dataset.lower() == 'svhn':
        module = CIFAR10net
    else:
        raise ValueError("invalid dataset name: %s" % dataset)
    return module


@ex.capture
def train(_run, layer_name, lr, mu, lambda_bar, batch_size, device,
          max_epochs, epochs_update,
          alphaupdate, optimizer):
    module = get_module()
    X, y = get_dataset()

    alphaupdate_kwags = flatten_dict(alphaupdate,
                                     preffix='callbacks__AlphaUpdate')

    checkpoint_tmpfile = '/tmp/weights.pt'
    callbacks = [LRScheduler(policy='StepLR', gamma=0.1, step_size=200)]

    net = LDMnet(module,
                 criterion=torch.nn.CrossEntropyLoss,
                 layer_name=layer_name,
                 mu=mu,
                 lambda_bar=lambda_bar,
                 lr=lr,
                 epochs_update=epochs_update,
                 max_epochs=max_epochs,
                 batch_size=batch_size,
                 device=device,
                 callbacks=callbacks,
                 **alphaupdate_kwags,
                 **flatten_dict(optimizer, 'optimizer'))

    net.fit(X, y=y)
    net.save_params(checkpoint_tmpfile)
    ex.add_artifact(checkpoint_tmpfile)

@ex.automain
def main(seed):
    train()
