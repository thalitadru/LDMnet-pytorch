#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:59:06 2018

@author: thalita
"""

import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def knn_distance_matrix(X, n_neighbors=10, nn_radius='halfk', leaf_size=30):
    knn = NearestNeighbors(n_neighbors, algorithm='auto', metric='sqeuclidean',
                           leaf_size=leaf_size, n_jobs=-1)
    knn.fit(X)
    W = knn.kneighbors_graph(n_neighbors=n_neighbors, mode='distance')

    if nn_radius == 'halfk':
        nn_radius = n_neighbors//2

    distances, _ = knn.kneighbors(n_neighbors=nn_radius)
    half_k_neighbors_distance = np.sqrt(distances[:, -1].squeeze())
    # normalization based on each points "neighbohood radius"
    for i in range(W.shape[0]):
        W[i,:] /= half_k_neighbors_distance[i]
    W = W.tocsc()
    for j in range(W.shape[0]):
         W[:,j] /= half_k_neighbors_distance[i]

    return W

def compute_W(X, n_neighbors=10, nn_radius='halfk', leaf_size=30, y=None):
    if y is None:
        W = knn_distance_matrix(X, n_neighbors=n_neighbors,
                                nn_radius=nn_radius,
                                leaf_size=leaf_size)
    else:
        classes = set(y)
        W = sp.lil_matrix((X.shape[0],X.shape[0]))
        for class_id in classes:
            idx = np.where(y==class_id)[0]
            class_W = knn_distance_matrix(X[idx], n_neighbors=n_neighbors,
                                          nn_radius=nn_radius,
                                          leaf_size=leaf_size)
            for j, i in enumerate(idx):
                W[idx,i] = class_W[:,j]
        W = W.tocsc()
    # comput exp over non zero entries (zero entries are distances non calculated)
    W *= -1
    W = W.expm1()
    W[W.nonzero()] += 1 # operation modifies W in place
    W.eliminate_zeros()
    return W


def ckech_connectivity(W):
    return nx.algorithms.is_connected(nx.from_scipy_sparse_matrix(W))


def compute_L(W, normalized=False, signless=False, return_diag=False):
    dd = W.sum(axis=1).squeeze()
    if normalized:
        invD = sp.dia_matrix((np.sqrt(1/dd), 0), shape=W.shape)
        # Mutiplication by diagonal matrix = pointwise mutiplication
        term =  (invD*W*invD) if signless else (-invD*W*invD)
        L = sp.eye(W.shape[0]) + term
    else:
        D = sp.dia_matrix((dd, 0),shape=W.shape)
        L = (D + W) if signless else (D - W)
    if return_diag:
        return L, dd
    else:
        return L


def plot_graph(X, color, W):
    G = nx.from_scipy_sparse_matrix(W)
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    pos = lambda x: dict([(i,l) for i,l in enumerate(x)])
    nx.draw_networkx(G,with_labels=False, pos=pos(X), alpha=0.8,
                 node_size=8, node_color=color,
                 edgelist=edges, edge_color=weights,
                 edge_cmap=plt.cm.Greys, edge_vmin=-0.05, edge_vmax=1)
    plt.title('kNN graph from W')


