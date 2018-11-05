#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 18:57:00 2018

@author: thalita
"""

import torch
from torch import nn
import torch.nn.functional as F


class MNISTnet(nn.Module):
    """Module outputs last layer with no softmax or logsoftmax.
    Intended to be used with nn.CrossEntropyLoss"""
    def __init__(self, dropout=0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.conv3 = nn.Conv2d(50, 500, kernel_size=4)
        self.dropout = dropout
        self.output = nn.Linear(500 * 1 * 1, 10)

    def forward(self, X, name='output', **kwargs):
        x = self.pool(self.conv1(X))
        x = self.pool(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.output.in_features)
        if name == 'conv3':
            return x
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        return x


def test_MNISTnet_forward():
    img = torch.rand(5, 1, 28, 28)
    net = MNISTnet()
    out = net(img)
    assert out.shape == (5, 10)

    out = net(img, name='conv3')
    assert out.shape == (5, 500)
    return net


class CIFAR10net(nn.Module):
    """Module outputs last layer with no softmax or logsoftmax.
    Intended to be used with nn.CrossEntropyLoss.
    This network can be applied to any RGB 32x32 images.
    """
    def __init__(self, dropout=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.conv2 = nn.Conv2d(96, 128, kernel_size=3, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4)
        self.fc1 = nn.Linear(256 * 2 * 2, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.dropout = dropout
        self.output = nn.Linear(2048, 10)

    def forward(self, X, name='output', **kwargs):
        x = self.pool(F.relu(self.conv1(X)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten
        x = x.view(-1, self.fc1.in_features)

        x = F.relu(self.fc1(x))
        if name == 'fc1':
            return x
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        if name == 'fc2':
            return x
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x)
        return x


def test_CIFAR10net_forward():
    img = torch.rand(5,3,32,32)
    net = CIFAR10net()
    out = net(img)
    assert out.shape == (5,10)
    out = net(img, name='fc2')
    assert out.shape == (5, 2048)
    return net

