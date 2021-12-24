'''
December 22th 2021
Functions for dimensionality reduction.'''

from __future__ import division
import numpy as np
import torch
import numpy.linalg as la
from sklearn import decomposition, manifold


def run_dim_red(inp_data, params, inp_dim, method='iso', stabilize=True):
    # Variance stabilization option included, since we're usually
    # working with Poisson-like data
    if stabilize:
        data_to_use = np.sqrt(inp_data)
    else:
        data_to_use = inp_data.copy()
    if method == 'iso':
        iso_instance = manifold.Isomap(n_neighbors=params['n_neighbors'],
                                       n_components=params['target_dim'])
        proj_data = iso_instance.fit_transform(data_to_use)
    elif method == 'lle':
        lle_instance = manifold.LocallyLinearEmbedding(n_neighbors=params['n_neighbors'],
                                                       n_components=params['target_dim'])
        proj_data = lle_instance.fit_transform(data_to_use)

    elif method == 'tsne':
        tsne_instance = manifold.TSNE(n_components=params['target_dim'], early_exaggeration=2.0)
        proj_data = tsne_instance.fit_transform(data_to_use)

    return proj_data
