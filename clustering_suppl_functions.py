#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 18:45:16 2021

@author: polina
"""
import numpy as np
import matplotlib as mpl
from scanfc.clustering import FoldChanges

def scale_data(data, fc_unscaled, sd_scale=True, norm_scale=True):
    """
    Function performing pre-procesing of the omic data based on unscaled omic 
    fold changes and returns processed data that can be used to generate 
    scaled fold changes. 

    Parameters
    ----------
    data : ndarray
        4D array with the dimensions corresponding to:
        1) nb of time points, 2) two experimental conditions
        (dim 0: control, dim 1: case)), 3) replicates, 4) nb of entities.
    fc_unscaled : FoldChanges
        DESCRIPTION.
    sd_scale : bool, optional
        If True (default), scaling by standard deviation is applied.
    norm_scale : bool, optional
        If True (default), scaling by fold change norm is applied.

    Returns
    -------
    data_doublenorm : ndarray
        Same format as input data, but scaled.

    """
    fc_cov = fc_unscaled.cov
    fc_var = np.diagonal(fc_cov, axis1=1, axis2=2)
    fc_sd = np.sqrt(fc_var)
    gene_names = fc_unscaled.var_names
    time_pts = fc_unscaled.time_points
    
    ### Normalizing FCs with respect to std:
    if sd_scale:
        data_norm = (data.transpose((1,2,0,3)) / fc_sd).transpose((2,0,1,3))
        fc_norm = FoldChanges(data=data_norm, var_names=gene_names,
                              time_points=time_pts)
    else:
        fc_norm = FoldChanges(data=data, var_names=gene_names,
                              time_points=time_pts)
    # Adding total norm scaling:
    if norm_scale:
        dim_data = (data.shape[0] * data.shape[1] * data.shape[2],
                    data.shape[3])
        fc_norms = fc_norm.compute_fc_norms()
        fc_norm_reshaped = (np.repeat(fc_norms, dim_data[0])
                            .reshape((dim_data[1], dim_data[0])))
        fc_norm_reshaped = fc_norm_reshaped.T.reshape(data.shape)
        
        data_doublenorm = data_norm / fc_norm_reshaped
    else:
        data_doublenorm = data_norm.copy()
    return data_doublenorm
    
def reorder_clusters(new_order, fc_clusters, fc_centroids):
    """
    Reorders fold changes clusters and corresponding centroids with respect

    Parameters
    ----------
    new_order : array-like
        1D array-like of length equal to the number of clusters, a permutation 
        of the original order of clusters.
    fc_clusters : ndarray
        1D array containing integers indicating clusters
        to which the fold changes are assigned.
    fc_centroids : ndarray
        1D array of length k containing indices in range (0, nb_var) of
        the fold changes that act as centroids.

    Returns
    -------
    fc_clusters_new_order : ndarray
        Same as fc_clusters but labels changed with respect to the new order.
    fc_centroids_new_order : ndarray
        Same as fc_centroids but reordered with respect to the new order.

    """
    nb_var = len(fc_clusters)
    fc_clusters_new_order = np.zeros(nb_var, dtype=int)
    for i, no in enumerate(new_order):
        fc_clusters_new_order = np.where(fc_clusters==no, i, fc_clusters_new_order)
    fc_centroids_new_order = fc_centroids[np.array(new_order)]
    return fc_clusters_new_order, fc_centroids_new_order