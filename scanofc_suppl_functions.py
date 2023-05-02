#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 18:45:16 2021

@author: polina
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles
from scanofc import FoldChanges, Clustering, NetworkInference
from sklearn.metrics.cluster import contingency_matrix
mpl.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'
#%matplotlib qt   

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
    
def joint_clustering_analysis(clusters_1, clusters_2, clusters_3, var_names,
                              print_result=False):
    """
    Performs joint clustering analysis produced by three methods (one 
    reference method and two alternative ones) by matching the results and 
    futher classifying elements in each cluster with respect to the 
    consistency of the clustreing results. Clusters with the same label 
    for different methods should correspond to each other (require being 
    previously reordered in some cases).

    Parameters
    ----------
    clusters_1 : ndarray
        1D array containing integers indicating clusters
        to which the fold changes are assigned by the first method (reference).
    clusters_2 : ndarray
        1D array containing integers indicating clusters
        to which the fold changes are assigned by the second method 
        (first alternative).
    clusters_3 : ndarray
        1D array containing integers indicating clusters
        to which the fold changes are assigned by the third method 
        (second alternative).
    var_names : array-like
        1D array-like containing data with `string` type, representing names
        of the considered entities.
    print_result : bool, optional
        If True, the results of the analysis are printed. The default is False.

    Returns
    -------
    Pandas DataFrame
        Columns represent original clusters, and rows represent new 
        classification by consistency order: 
            - 'Order 1': elements that are put in the considered cluster by all
            3 methods;
            - 'Order 2': elements that are put in the considered cluster by 
            the reference method and one alternative;
            - 'Order 3': elements that are put in the considered cluster only
            by the reference method.

    """
    nb_cl = len(np.unique(clusters_1))
    same_nb_cl = (nb_cl==len(np.unique(clusters_2)))&(nb_cl==len(np.unique(clusters_3)))
    assert same_nb_cl, 'Error: different number of clusters detected'
    res_dict = {}
    for i in range(nb_cl):
        cluster_i_in_1 = (clusters_1==i).nonzero()[0]
        cluster_i_in_2 = (clusters_2==i).nonzero()[0]
        int_1_2 = np.intersect1d(cluster_i_in_1, cluster_i_in_2)
        cluster_i_in_3 = (clusters_3==i).nonzero()[0]
        int_1_3 = np.intersect1d(cluster_i_in_1, cluster_i_in_3)
        cluster_i_in_all = np.intersect1d(int_1_2, cluster_i_in_3)
        genes_order_1 = var_names[cluster_i_in_all]
        res_dict[f'Cluster {i+1}'] = {}
        res_dict[f'Cluster {i+1}']['Order 1'] = var_names[cluster_i_in_all]
        cluster_i_in_1_n_2_or_3 = np.union1d(int_1_2, int_1_3)
        genes_order_2 = var_names[np.setdiff1d(cluster_i_in_1_n_2_or_3, cluster_i_in_all)]
        res_dict[f'Cluster {i+1}']['Order 2'] = genes_order_2
        genes_order_3 = var_names[np.setdiff1d(cluster_i_in_1, cluster_i_in_1_n_2_or_3)]
        res_dict[f'Cluster {i+1}']['Order 3'] = genes_order_3
        if print_result:
            print(f"Cluster {i+1}: ")
            print("Order 1: ")
            print('   '.join(genes_order_1))
            print("Order 2:")
            print('   '.join(genes_order_2))
            print("Order 3:")
            print('   '.join(genes_order_3))
            print("\n")
    return pd.DataFrame(res_dict)

def compare_clusters(clusters_1, clusters_2, var_names, print_result=False):
    """
    Compares two results of clustering produced for two conditions by 
    identifying intersections and differences for each cluster.

    Parameters
    ----------
    clusters_1 : ndarray
        1D array containing integers indicating clusters
        to which the fold changes are assigned for the first condition.
    clusters_2 : ndarray
        1D array containing integers indicating clusters
        to which the fold changes are assigned for the second condition.
    var_names : array-like
        1D array-like containing data with `string` type, representing names
        of the considered entities.
    print_result : bool, optional
        If True, the results of the analysis are printed. The default is False.

    Returns
    -------
    Pandas DataFrame
        Columns represent original clusters, and rows represent new 
        classification produced by matching two conditions: 
            -'Common': elements that are put in the considered cluster for 
            both conditions;
            -'Condition 1': elements that are put in the considered cluster 
            only for the first condition;
            -'Condition 2': elements that are put in the considered cluster 
            only for the second condition.

    """
    nb_cl = clusters_1.max() + 1
    res_dict = {}
    for i in range(nb_cl):
        cl_i_c1 = (clusters_1==i).nonzero()[0]
        cl_i_c2 = (clusters_2==i).nonzero()[0]
        common = var_names[np.intersect1d(cl_i_c1, cl_i_c2)]
        res_dict[f'Cluster {i+1}'] = {}
        res_dict[f'Cluster {i+1}']['Common'] = common
        c1 = var_names[np.setdiff1d(cl_i_c1, cl_i_c2)]
        res_dict[f'Cluster {i+1}']['Condition 1'] = c1
        c2 = var_names[np.setdiff1d(cl_i_c2, cl_i_c1)]
        res_dict[f'Cluster {i+1}']['Condition 2'] = c2
        if print_result:
            print(f"Cluster {i+1}:")
            print("Common:")
            print('   '.join(common))
            print("Condition 1")
            print('   '.join(c1))
            print("Condition 2")
            print('   '.join(c2))
            print("\n")
    return pd.DataFrame(res_dict)

def plot_cluster_venn_summary(jca_cond1, jca_cond2, compclust,
                              condlabel_1='1', condlabel_2='2',
                              figsize=(20,15), fontsize=12):
    """
    Plots k Venn diagrams where k is the number of clusters in all considered
    clusterings that illustrate comparative clustering produced for two 
    conditions while taking into account three clustering methods. Each Venn 
    diagram contains lists of elements placed with respect to whether they are 
    clustered correspondingly for both conditions or not, and colored with 
    respect to the consistency of clustering by the three methods: red if 
    order 1, blue if order 2, and black if order 3. In case of the intersection,
    matching orders becomes ambiguous, and colors are assigned as follows: red
    if order 1 for both conditions, 2 if order 1 for only one condition or 
    order 2 for both, and black if order 2 for only one condition or order 3 
    for both.

    Parameters
    ----------
    jca_cond1 : Pandas DataFrame
        Produced by the function 'joint_clustering_analysis' for condition 1.
    jca_cond2 : Pandas DataFrame
        Produced by the function 'joint_clustering_analysis' for condition 2.
    compclust : Pandas DataFrame
        Produced by the function 'compare_clusters' for the two considered 
        conditions.
    condlabel_1 : string, optional
        Label for the first condition. The default is '1'.
    condlabel_2 : string, optional
        Label for the first condition. The default is '2'.
    figsize : (float, float), optional
        Width and height of the figure(s). The default is (20,15).
    fontsize : float, optional
        Baseline fontsize for the text on the diagrams. The default is 12.

    Returns
    -------
    None

    """
    def join_str(list_str, prop):
        if prop > 0.5:
            text = ''
            for c, w in enumerate(list_str):
                text += w
                text += '\n' if (c + 1) % 3 == 0 else '  '
        elif prop > 0.25:
            text = ''
            for c, w in enumerate(list_str):
                text += w
                text += '\n' if (c +1) % 2 == 0 else '  '
        else:
            text = '\n'.join(list_str)
        return text
        
    nb_cl = len(compclust.columns)
    for i in range(1,nb_cl+1):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f'Cluster {i}', fontsize=fontsize * 2, font='serif')
        venn_dict = {'10': compclust[f'Cluster {i}']['Condition 1'].size,
                     '01': compclust[f'Cluster {i}']['Condition 2'].size,
                     '11': compclust[f'Cluster {i}']['Common'].size}
        v2 = venn2(subsets = venn_dict,
                   set_labels=(condlabel_1, condlabel_2))
        venn2_circles(subsets = venn_dict, linestyle='dashed',
                      linewidth=3, color='silver');
        for l in v2.set_labels:
            l.set_font('serif')
            l.set_fontsize(fontsize * 2)
        if venn_dict['10']>0: v2.get_patch_by_id('10').set_color('#fe7b7c')
        if venn_dict['01']>0: v2.get_patch_by_id('01').set_color('#12e193')
        if venn_dict['11']>0: v2.get_patch_by_id('11').set_color('#887191')

        ax = plt.gca()
        # Condition 1:
        if venn_dict['10']>0: 
            v2.get_label_by_id('10').set_text('')
            x = v2.get_label_by_id('10')._x
            y = v2.get_label_by_id('10')._y
            c1_o1 = np.intersect1d(compclust[f'Cluster {i}']['Condition 1'],
                                   jca_cond1[f'Cluster {i}']['Order 1'])
            c1_o2 = np.intersect1d(compclust[f'Cluster {i}']['Condition 1'],
                                   jca_cond1[f'Cluster {i}']['Order 2'])
            c1_o3 = np.intersect1d(compclust[f'Cluster {i}']['Condition 1'],
                                   jca_cond1[f'Cluster {i}']['Order 3'])
            prop10 = venn_dict['10']/sum(venn_dict.values())
            x_adj = (x - 0.1) if prop10 > 0.3 else x
            x_adj -= 0.01
            y_adj = ((y - 0.1) if (len(c1_o1)<=2 or len(c1_o1)+len(c1_o2)<=5) else y)
            y_adj += ((y - 0.1) if (len(c1_o3)<=2 or len(c1_o2)+len(c1_o3)<=5) else y)
            fsize = fontsize
            ygap = (fontsize * 1.5 if prop10 > 0.5 else fontsize * 1 if prop10 < 0.2
                    else fontsize * 1.2)
            ygap /= 100
            ax.text(x_adj, y_adj + ygap, join_str(c1_o1, prop10), color='r',
                    size=fsize, font='serif')
            ax.text(x_adj, y_adj, join_str(c1_o2, prop10), color='b',
                    size=fsize, font='serif')
            ax.text(x_adj, y_adj - ygap, join_str(c1_o3, prop10), color='k',
                    size=fsize, font='serif')
        
        # Condition 2:
        if venn_dict['01']>0:
            v2.get_label_by_id('01').set_text('')
            x = v2.get_label_by_id('01')._x
            y = v2.get_label_by_id('01')._y
            c2_o1 = np.intersect1d(compclust[f'Cluster {i}']['Condition 2'],
                                   jca_cond2[f'Cluster {i}']['Order 1'])
            c2_o2 = np.intersect1d(compclust[f'Cluster {i}']['Condition 2'],
                                   jca_cond2[f'Cluster {i}']['Order 2'])
            c2_o3 = np.intersect1d(compclust[f'Cluster {i}']['Condition 2'],
                                   jca_cond2[f'Cluster {i}']['Order 3'])
            prop01 = venn_dict['01']/sum(venn_dict.values())
            ygap = (fontsize * 1.5 if prop01 > 0.5 else fontsize * 1 if prop01 < 0.2
                    else fontsize * 1.2)
            ygap /= 100
            x_adj = (x - 0.15) if prop01 > 0.3 else x
            x_adj -= 0.05
            y_adj = ((y + 0.1) if (len(c2_o1)<=2 or len(c2_o1)+len(c2_o2)<=5) else y)
            y_adj += ((y - 0.1) if (len(c2_o3)<=2 or len(c2_o2)+len(c2_o3)<=5) else y)
            fsize = fontsize
            ax.text(x_adj, y_adj + ygap, join_str(c2_o1, prop01), color='r',
                    size=fsize, font='serif')
            ax.text(x_adj, y_adj, join_str(c2_o2, prop01), color='b',
                    size=fsize, font='serif')
            ax.text(x_adj, y_adj - ygap, join_str(c2_o3, prop01), color='k',
                    size=fsize, font='serif')
        
        # Common:
        if venn_dict['11']>0:
            v2.get_label_by_id('11').set_text('')
            x = v2.get_label_by_id('11')._x
            y = v2.get_label_by_id('11')._y
            int_c1_o1 = np.intersect1d(compclust[f'Cluster {i}']['Common'],
                                       jca_cond1[f'Cluster {i}']['Order 1'])
            int_c2_o1 = np.intersect1d(compclust[f'Cluster {i}']['Common'],
                                       jca_cond2[f'Cluster {i}']['Order 1'])
            int_o1 = np.intersect1d(int_c1_o1, int_c2_o1)
            int_c1_o1_rest = np.setdiff1d(int_c1_o1, int_o1)
            int_c2_o1_rest = np.setdiff1d(int_c2_o1, int_o1)
            int_o1_rest = np.union1d(int_c1_o1_rest, int_c2_o1_rest)
            int_c1_o2 = np.intersect1d(compclust[f'Cluster {i}']['Common'],
                                       jca_cond1[f'Cluster {i}']['Order 2'])
            int_c2_o2 = np.intersect1d(compclust[f'Cluster {i}']['Common'],
                                       jca_cond2[f'Cluster {i}']['Order 2'])
            int_o2 = np.setdiff1d(np.union1d(np.intersect1d(int_c1_o2, int_c2_o2),
                                             int_o1_rest), int_o1)
            int_c1_o2_rest = np.setdiff1d(int_c1_o2, int_o2)
            int_c2_o2_rest = np.setdiff1d(int_c2_o2, int_o2)
            int_o2_rest = np.union1d(int_c1_o2_rest, int_c2_o2_rest)
            int_c1_o3 = np.intersect1d(compclust[f'Cluster {i}']['Common'],
                                       jca_cond1[f'Cluster {i}']['Order 3'])
            int_c2_o3 = np.intersect1d(compclust[f'Cluster {i}']['Common'],
                                       jca_cond2[f'Cluster {i}']['Order 3'])
            already_in_set = np.union1d(int_o1, int_o2)
            int_o3 = np.setdiff1d(np.union1d(np.union1d(int_c1_o3, int_c2_o3),
                                             int_o2_rest), already_in_set)
            prop = venn_dict['11']/sum(venn_dict.values())
            ygap = (fontsize * 1.5 if prop > 0.5 else fontsize * 1 if prop < 0.2
                    else fontsize * 1.2)
            ygap /= 100
            x_adj = ((x - 0.1) if (prop > 0.3 and prop10 < 0.2) 
                     else (x - 0.1) if (prop > 0.3 and prop01 < 0.2) else x)
            y_adj = ((y + 0.1) if (len(int_o1)<=2 or len(int_o1)+len(int_o2)<=5) else y)
            y_adj += ((y - 0.1) if (len(int_o3)<=2 or len(int_o2)+len(int_o3)<=5) else y)
            fsize = fontsize
            ax.text(x_adj, y_adj + ygap, join_str(int_o1, prop), color='r',
                    size=fsize, font='serif')
            ax.text(x_adj, y_adj, join_str(int_o2, prop), color='b',
                    size=fsize, font='serif')
            ax.text(x_adj, y_adj - ygap, join_str(int_o3, prop), color='k',
                    size=fsize, font='serif')
        plt.show()