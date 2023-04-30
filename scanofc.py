"""
ScanOFC : Statistical framework for Clustering with Alignment and
    Network inference of Omic Fold Changes.

@author: Polina Arsenteva

"""
import itertools
import warnings
import numpy as np
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import pandas as pd
from sklearn.datasets import make_spd_matrix as spd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import umap
import seaborn as sns
from sparsebm import SBM
import networkx as nx
mpl.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'


class FoldChanges():
    """
    A class representing a set of fold changes (a measure of difference between
    the two experimental conditions over time).

    Attributes
    ----------
    means : ndarray
        2D array of shape (nb_time_points, nb_var) containing data
        with `float` type, representing fold changes' means for each entity
        and each time point.
    cov : ndarray
        3D array of shape (nb_time_pts, nb_var, nb_var) containing
        data with `float` type, representing fold changes' nb_var x nb_var
        shaped covariance matrices for each time point. Time-wise
        cross-covariances are assumed to be 0 due to experimental design.
        In case of Hellinger distance, can also be 4-dimensional (natural form):
        (nb_time_pts, nb_time_pts, nb_var, nb_var).
    sd : ndarray
        2D array of shape (nb_time_pts, nb_var) containing data
        with `float` type, representing fold changes' standard deviation for
        each entity and each time point.
    time_points : array-like
        1D array-like containing data with `float` type, representing time
        points at which fold changes were measured. If not given then range of
        indices of the corresponding dimension.
    nb_time_pts : int
        number of time points measured, i.e. len(time_points) or the size of
        the corresponding dimension.
    var_names : array-like or None
        1D array-like containing data with `string` type, representing names
        of the measured entities (ex. genes). If not given then range of
        indices of the corresponding dimension of means.
    nb_var : int
        number of entities considered, i.e. len(var_names) or the size of
        the corresponding dimension of means.

    Methods
    -------
    compute_distance_pairs(dist='d2hat', sign_pen=False, pen_param=10)
        Computes fold changes' pairwise distances.
    compute_fc_norms(dist='d2hat')
        Computes fold changes' norms with respect to the chosen distance.
    compute_dist_mat(index_pairs, distances)
        Transforms the set of pairwise distances into a distance matrix.
    compute_cross_distances(bary_means, bary_cov, cluster=None)
        Calculates the Wasserstein distance in different configurations.
    compute_warped_distance_pairs(max_warp_step=1, sign_pen=False,
                                  pen_param=0.01)
        Computes fold changes' pairwise distances for all considered warps.
    compute_warped_dist_mat(index_pairs, warped_distances)
        Calculates an optimal warping distance matrix and an optimal warp
        matrix from the set of pairwise warped distances.

    """

    def __init__(self, data=None, means=None, cov=None,
                 var_names=None, time_points=None):
        """
        Parameters
        ----------
        data : ndarray or None
            If not None, 4D array with the dimensions corresponding to:
            1) nb of time points, 2) two experimental conditions
            (dim 0: control, dim 1: case)), 3) replicates, 4) nb of entities.
            If None (by default), then the fold changes are constructed based
            on 'means' and 'cov'. Either 'data' or 'means' and 'cov' have to be
            non-None, with 'data' having priority for the fold changes
            construction.
        means : ndarray or None
            If not None, 2D array of shape (nb_time_pts, nb_var)
            containing data with `float` type, representing fold changes' means
            for each entity and each time point. If 'data' is None, used to
            construct fold changes. Either 'data' or 'means' and 'cov' have to
            be non-None.
        cov : ndarray or None
            If not None, 3D array of shape (nb_time_pts, nb_var, nb_var)
            containing data with `float` type, representing fold changes'
            nb_var x nb_var shaped covariance matrices for each time point.
            Time-wise cross-covariances are assumes to be 0 due to experimental
            design. In case of Hellinger distance, can also be 4-dimensional
            (natural form): (nb_time_pts, nb_time_pts, nb_var, nb_var).
            If 'data' is None, used to construct fold changes. Either 'data' or
            'means' and 'cov' have to be non-None.
        var_names : array-like or None
            If not None, 1D array-like containing data with `string` type,
            representing names of the measured entities (ex. genes).
            The default is None.
        time_points : array-like or None
            If not None, 1D array-like containing data with `float` type,
            representing time points at which fold changes were measured.
            The default is None.

        """
        assert_str = "Either 'data' or 'means' and 'cov' have to be non-None"
        assert (data is not None) or (
            means is not None and cov is not None), assert_str
        if data is not None:
            dim_data = data.shape
            means = np.nanmean(data, axis=2)
            covmat = np.zeros(
                (dim_data[0], dim_data[1], dim_data[3], dim_data[3]))
            for t in range(dim_data[0]):
                for c in range(dim_data[1]):
                    # Masking missing replicates:
                    masked_covmat = np.ma.masked_invalid(data[t, c])
                    covmat[t, c, :, :] = np.ma.getdata(np.ma.cov(masked_covmat,
                                                                 rowvar=False))
            # Fold changes = case - control
            fc_means = means[:, 1, :] - means[:, 0, :]
            fc_cov = covmat[:, 1, :, :] + covmat[:, 0, :, :]
            self.means = fc_means
            self.cov = fc_cov
        else:
            self.means = means
            self.cov = cov
        self.sd = np.sqrt(np.diagonal(self.cov, axis1=1, axis2=2))
        self.nb_var = (len(var_names) if var_names is not None
                       else self.means.shape[1])
        self.nb_time_pts = (len(time_points) if time_points is not None
                            else self.means.shape[0])
        self.time_points = (time_points if time_points is not None
                            else np.arange(self.nb_time_pts))
        self.var_names = (var_names if var_names is not None
                          else np.arange(self.nb_var).astype(str))

    def compute_distance_pairs(self, dist='d2hat', sign_pen=False, pen_param=10):
        """
        Computes pairwise distances for a set of fold changes encoded in
        FoldChanges class instance for a chosen distance. The choice of a
        distance is limited to L2 distance between random fold changes'
        estimators, and Hellinger distance.

        Parameters
        ----------
        dist : str
            Can be either 'd2hat' (L2 distance between random estimators,
            default), or 'hellinger' (Hellinger distance).
        sign_pen : bool
            True if sign penalty should be added to the distance, False
            otherwise (default). Sign penalty penalizes fold changes pairs that
            have different signs in one or more time points.
        pen_param : float
            Sign penalty hyperparameter (weight of penalty).

        Returns
        -------
        index_pairs : ndarray
            2D array of shape (number of pairs, 2) containing pairs
            of indices in the same order as the pairwise distances array
            'distances'.
        distances : ndarray
            1D array of length equal to the number of pairs containing
            pairs of distances between the fold changes in the same order as
            'index_pairs'.

        """
        # For K-medoid and initialization
        index_pairs = np.array(
            list(itertools.combinations(range(self.nb_var), 2)))
        all_comb_means = self.means[:, index_pairs]
        mu_diff = (all_comb_means[:, :, 0] - all_comb_means[:, :, 1])
        if dist == 'hellinger':
            if len(self.cov.shape) == 3:  # implied by the experimental setting
                cov_4d = np.zeros((self.nb_time_pts, self.nb_time_pts,
                                   self.nb_var, self.nb_var))
                cov_4d[np.diag_indices(self.nb_time_pts)] = self.cov.copy()
            if len(self.cov.shape) == 4:  # natural for Hellinger distance
                cov_4d = self.cov.copy()
            sigma1 = cov_4d[:, :, index_pairs[:, 0], index_pairs[:, 0]]
            sigma2 = cov_4d[:, :, index_pairs[:, 1], index_pairs[:, 1]]
            sigma_mean = (sigma1 + sigma2)/2
            det_sigma1 = np.linalg.det(sigma1.transpose((2, 0, 1)))
            det_sigma2 = np.linalg.det(sigma2.transpose((2, 0, 1)))
            det_mean = np.linalg.det(sigma_mean.transpose((2, 0, 1)))
            det_term = (det_sigma1**(1/4)) * \
                (det_sigma2**(1/4)) / (det_mean**(1/2))
            prod1 = np.einsum('ij,jik->jk', mu_diff,
                              np.linalg.inv(sigma_mean.transpose((2, 0, 1))))
            exp_term = (-1/8) * np.einsum('ij,ji->i', prod1, mu_diff)
            distances = 1 - det_term * np.exp(exp_term)
        else:
            norm_means = np.sum(mu_diff**2, axis=0)
            sigma1 = self.cov[:, index_pairs[:, 0], index_pairs[:, 0]]
            sigma2 = self.cov[:, index_pairs[:, 1], index_pairs[:, 1]]
            sigma12 = self.cov[:, index_pairs[:, 0], index_pairs[:, 1]]
            distances = norm_means + np.sum(sigma1, axis=0) + np.sum(sigma2, axis=0) \
                - 2*np.sum(sigma12, axis=0)
            if sign_pen:
                product = all_comb_means[:, :, 0] * all_comb_means[:, :, 1]
                penalty = (pen_param * np.sign(product).sum(axis=0)
                           / self.nb_time_pts)
                distances -= penalty - penalty.max()
        return index_pairs, distances

    def compute_fc_norms(self, dist='d2hat'):
        """
        Computes norms for all fold changes in the class instance with respect
        to the chosen distance (so far implemented only for the L2 distance
        between random estimators).

        Parameters
        ----------
        dist : str
            So far 'd2hat' is the default and the only option.

        Returns
        -------
        ndarray
            1D array of length equal to the number of fold changes (that is,
            the number of biological entities considered) containing the norms.

        """
        # For K-medoid and initialization
        fc_norms = np.sum(self.means**2, axis=0)
        fc_var = np.diagonal(self.cov, axis1=1, axis2=2)
        if dist == 'd2hat':
            fc_norms += np.sum(fc_var, axis=0)
        return np.sqrt(fc_norms)

    def compute_dist_mat(self, index_pairs, distances):
        """
        Transforms the set of pairwise distances into a distance matrix.

        Parameters
        ----------
        index_pairs : ndarray
            2D array of shape (number of pairs, 2) containing pairs
            of indices in the same order as the pairwise distances array
            'distances'.
        distances : ndarray
            1D array of length equal to the number of pairs containing
            pairs of distances between the fold changes in the same order as
            'index_pairs'.

        Returns
        -------
        dist_mat : ndarray
            Distance matrix, 2D array of shape with both dimensions
            equal to the number of fold changes (entities).

        """
        # For K-medoid and the comparative cost function
        self.nb_var = np.max(index_pairs) + 1
        dist_mat = np.zeros((self.nb_var, self.nb_var))
        dist_mat[tuple(index_pairs.T)] = distances
        dist_mat[tuple(index_pairs[:, (1, 0)].T)] = distances
        return dist_mat

    @staticmethod
    def _sqrt_mat(m):
        """
        Computes a square root of a matrix by diagonalization, used in
        '_compute_K' to calculate the matrix K that specifies the joint
        distribution of two fold changes when calculating Wasserstein distance.
        Distinguishes three cases with different dimensions in different parts
        of the function '_compute_K'.

        Parameters
        ----------
        m : ndarray
            2D, 3D or 4D array to take a square root of. In all cases, square
            root is taken with respect to the last to dimensions, which should
            be square and diagonalizable.

        Returns
        -------
        ndarray
            Square root of m.

        """
        eig_val, eig_vec = np.linalg.eigh(m)
        #eig_val, eig_vec = np.linalg.eig(m)
        # Simple matrix case:
        if len(m.shape) == 2:
            return eig_vec @ np.diag(np.sqrt(eig_val)) @ (eig_vec.T)
        # Applied in _compute_K, inner sqrt:
        if len(m.shape) == 3:
            nb_cl = m.shape[0]
            timelen = m.shape[1]
            # Creating diagonal identity tensor with eigenvalues:
            id_tensor = (np.repeat(np.identity(timelen), nb_cl, axis=1)
                         .reshape((timelen, timelen, nb_cl)))
            # Matrix multiplications with respect to time dimensions, preserving
            # variables' dimension:
            eig_val_diag = np.einsum('ijk,ik->ijk',
                                     np.transpose(id_tensor, (2, 0, 1)),
                                     (np.sqrt(eig_val)))
            matprod = np.einsum('ijk,ikl->ijl', eig_vec, eig_val_diag)
            return np.einsum('ijk,ikl->ijl', matprod, np.transpose(eig_vec, (0, 2, 1)))
        # Applied in _compute_K, outer sqrt:
        if len(m.shape) == 4:
            nb_var = m.shape[1]
            nb_cl = m.shape[0]
            timelen = m.shape[2]
            # Creating diagonal identity tensor with eigenvalues:
            id_tensor = (np.repeat(np.identity(timelen), nb_var*nb_cl, axis=1)
                         .reshape((timelen, timelen, nb_var, nb_cl)))
            # Matrix multiplications with respect to time dimensions, preserving
            # variables' and clusters' dimensions:
            eig_val_diag = np.einsum('ijkl,ijk->ijkl',
                                     np.transpose(id_tensor, (3, 2, 0, 1)),
                                     (np.sqrt(eig_val)))
            matprod = np.einsum('ijkl,ijlm->ijkm', eig_vec, eig_val_diag)
            return np.einsum('ijkl,ijlm->ijkm', matprod, np.transpose(eig_vec,
                                                                      (0, 1, 3, 2)))

    @staticmethod
    def _compute_K(cov1, cov2):
        """
        Calculates the matrix K that specifies the joint distribution of two
        fold changes when calculating the Wasserstein distance.

        Parameters
        ----------
        cov1 : ndarray
            Covariance matrix, or a set of covariance matrices, that represent
            the first marginal distribution, of either all fold changes or in
            the cluster. See 'compute_cross_distances' for more details.
        cov2 : ndarray
            Covariance matrix, or a set of covariance matrices, that represent
            the second marginal distribution, of either all fold changes or
            the barycenter. See 'bary_cov' in 'compute_cross_distances'
            for more details.

        Returns
        -------
        K : ndarray
            Matrix characterizing the joint distributions of random variables
            with marginals 'cov1' and 'cov2'. For more details on different
            use cases see 'compute_cross_distances'.

        """
        # Simple matrix case:
        if len(cov2.shape) == 2:
            M_sqrt = FoldChanges._sqrt_mat(cov2)
            K = FoldChanges._sqrt_mat(M_sqrt @ cov1 @ M_sqrt)
        # 3D version to compute distances for a cluster simultaneously:
        if len(cov2.shape) == 3:
            cov2_reshaped = np.transpose(cov2, (2, 0, 1))
            M_sqrt = FoldChanges._sqrt_mat(cov2_reshaped)
            # Matrix multiplications with respect to time dimensions,
            # preserving variables' (and clusters') dimension(s):
            matprod = np.einsum('ijk,lkm->ijml', M_sqrt, cov1)
            tmp_mat = np.einsum('ijkl,ikm->ijml', matprod, M_sqrt)
            K = FoldChanges._sqrt_mat(np.transpose(tmp_mat, (0, 3, 1, 2)))
        return K

    def compute_cross_distances(self, bary_means, bary_cov, cluster=None):
        """
        Designed for the vectorized version of the Wasserstein k-means,
        in particular:
            - case 1: to compute distances between the fold changes in one
        cluster and the barycenter to update barycenter in function
        'compute_barycenter' of the Clustering class.
            - case 2: to compute distances between the fold changes
        and their barycenters in all clusters to assess cost in function
        'choose_k_clusters'of the Clustering class.
        In addition, it is used to compute pairwise
        distance matrix for the Wasserstein distance when instantiating a
        Clustering class (case 3, not used in k-means).

        Parameters
        ----------
        bary_means : ndarray
            Case 1: 2D array of shape (nb_time_pts, 1) containing the mean of
                    the current barycenter in the fixed point iteration.
            Case 2: 2D array of shape (nb_time_pts, k), where k stands for the
                    number of clusters (and hence barycenters). The array
                    contains the means of all barycenters in the current
                    iteration of k-means.
            Case 3: 2D array of shape (nb_time_pts, nb_var) containing
                    fold changes' means.
        bary_cov : ndarray
            Case 1: 2D array of shape (nb_time_pts, nb_time_pts) containing
                    the covariance matrix of the current barycenter in the
                    fixed point iteration.
            Case 2: 3D array of shape (nb_time_pts, nb_time_pts, k), where k
                    stands for the number of clusters (and hence barycenters).
                    The array contains the covariance matrices of all
                    barycenters in the current iteration of k-means.
            Case 3: 3D array of shape (nb_time_pts, nb_time_pts, nb_var)
                    containing marginal (diagonal) covariance matrices of the
                    fold changes.

        cluster : ndarray, optional
            None in cases 2 and 3, in case 1: 1D array of length equal to the
            size of the considered cluster. Contains data of type 'int'
            corresponding to the indices of the fold changes' belonging to
            this cluster.

        Returns
        -------
        wass_dist : ndarray
            Case 1: 2D array of shape (1, cluster_size) containing distances
                    between the fold changes in the cluster and the barycenter.
            Case 2: 2D array of shape (k, nb_var) containing distances between
                    the fold changes and the barycenters for all clusters.
            Case 3: 2D array of shape (nb_var, nb_var) containing pairwise
                    distances between the fold changes.

        K : ndarray
            Case 1: 4D array of shape (1, cluster_size, nb_time_pts, nb_time_pts)
                    characterizing the joint distributions of the fold changes
                    in the cluster and the barycenter. Central term in the
                    fixed point equation.
            Case 2: 4D array of shape (k, nb_var, nb_time_pts, nb_time_pts)
                    characterizing the joint distributions of the fold changes
                    and all the barycenters.
            Case 3: 4D array of shape (nb_var, nb_var, nb_time_pts, nb_time_pts)
                    characterizing the joint distributions of all
                    fold changes pairs.

        """

        if cluster is None:
            fc_means = self.means.copy()
            fc_cov = self.cov.copy()
        else:
            fc_means = self.means[:, cluster].copy()
            fc_cov = self.cov[:, cluster][:, :, cluster].copy()
        nb_cl = bary_means.shape[1]
        fc_var = np.diagonal(fc_cov, axis1=1, axis2=2)
        # Creating diagonal variance tensor:
        dim2 = fc_means.shape[1]
        id_tensor = (np.repeat(np.identity(self.nb_time_pts), dim2, axis=1)
                     .reshape((self.nb_time_pts, self.nb_time_pts, dim2)))
        M1 = np.einsum('ijk,ik->ijk', id_tensor, fc_var)
        M1 = np.transpose(M1, (2, 0, 1))
        K = self._compute_K(M1, bary_cov)
        fc_means_rep = (np.repeat(fc_means, nb_cl, axis=1)
                        .reshape(self.nb_time_pts, dim2, nb_cl))
        fc_means_rep = np.transpose(fc_means_rep, (0, 2, 1))
        bary_means_rep = (np.repeat(bary_means, dim2, axis=1)
                          .reshape(self.nb_time_pts, nb_cl, dim2))
        wass_dist = np.sum((fc_means_rep - bary_means_rep)**2, axis=0)
        tmp = ((np.repeat(np.trace(M1, axis1=1, axis2=2), nb_cl)
                .reshape(dim2, nb_cl).T)
               + np.repeat(np.trace(bary_cov), dim2).reshape(nb_cl, dim2)
               - 2*np.trace(K, axis1=2, axis2=3))
        wass_dist += tmp
        return wass_dist, K

    def compute_warped_distance_pairs(self, max_warp_step=1, sign_pen=False,
                                      pen_param=0.01):
        """
        Computes pairwise distances for a set of considered warps for a set of
        fold changes encoded in FoldChanges class instance for a chosen
        distance. Warped distances (or distances after alignment) are only
        calculated for the L2 distance between random fold changes' estimators.

        Parameters
        ----------
        max_warp_step : int
            If max_warp_step=i>0, then the set of all considered warps is the
            set of all integers between -i and i.
        sign_pen : bool
            True if sign penalty should be added to the distance, False
            otherwise (default). Sign penalty penalizes fold changes pairs that
            have different signs in one or more time points.
        pen_param : float
            Sign penalty hyperparameter (weight of penalty).

        Returns
        -------
        index_pairs : ndarray
            2D array of shape (number of pairs, 2) containing pairs
            of indices in the same order as the pairwise distances array
            'distances'.
        warped_distances : ndarray
            2D array such that the first dimension is of size
            2 * max_warp_step + 1 corresponding to all considered warps, and
            the second dimension corresponds to the number of fold changes
            pairs in the same order as 'index_pairs'.

        """
        # For K-medoid and initialization
        index_pairs = np.array(
            list(itertools.combinations(range(self.nb_var), 2)))
        all_comb_means = self.means[:, index_pairs]
        warped_distances = np.zeros((2*max_warp_step+1, index_pairs.shape[0]))
        for i in range(-max_warp_step, max_warp_step+1):
            warped_range_1 = np.asarray(
                range(-min(0, i), self.nb_time_pts-max(0, i)))
            warped_range_2 = np.asarray(
                range(max(0, i), self.nb_time_pts+min(0, i)))
            warped_range_12 = np.asarray(
                range(abs(i), self.nb_time_pts-abs(i)))
            warped_mu_diff = (all_comb_means[warped_range_1, :, 0]
                              - all_comb_means[warped_range_2, :, 1])
            warped_norm_means = np.sum(warped_mu_diff**2, axis=0)
            warped_sigma1 = self.cov[warped_range_1[:, np.newaxis],
                                     index_pairs[:, 0], index_pairs[:, 0]]
            warped_sigma2 = self.cov[warped_range_2[:, np.newaxis],
                                     index_pairs[:, 1], index_pairs[:, 1]]
            warped_sigma12 = self.cov[warped_range_12[:, np.newaxis],
                                      index_pairs[:, 0], index_pairs[:, 1]]
            warped_distances_i = (warped_norm_means
                                  + np.sum(warped_sigma1, axis=0)
                                  + np.sum(warped_sigma2, axis=0)
                                  - 2*np.sum(warped_sigma12, axis=0))
            warped_distances[max_warp_step+i] = (warped_distances_i
                                                 / (self.nb_time_pts - abs(i)))
            if sign_pen:
                product = (all_comb_means[warped_range_1, :, 0]
                           * all_comb_means[warped_range_2, :, 1])
                penalty = (pen_param * np.sign(product).sum(axis=0)
                           / (self.nb_time_pts - abs(i)))
                warped_distances[max_warp_step+i] -= penalty - pen_param
        return index_pairs, warped_distances

    def compute_warped_dist_mat(self, index_pairs, warped_distances):
        """
        Calculates an optimal warping distance matrix and an optimal warp
        matrix from the set of pairwise warped distances.

        Parameters
        ----------
        index_pairs : ndarray
            2D array of shape (number of pairs, 2) containing pairs
            of indices in the same order as the pairwise optimal warping
            distances array 'warped_distances' (with respect to the second
            dimension).
        warped_distances : ndarray
            2D array, the first dimension corresponds to all considered warp
            steps, the second corresponds to pairs of warped distances between
            the fold changes in the same order as 'index_pairs'.

        Returns
        -------
        warped_dist_mat : ndarray
            Optimal Warping Distance matrix, 2D array with both dimensions
            equal to the number of fold changes (entities). Distances are such
            that minimize the pairwise distance over the set of all considered
            warps.
        optimal_warp_mat : ndarray
            Optimal Warp matrix, 2D array with both dimensions equal to the
            number of fold changes (entities). The values of the upper
            triangular part of the matrix correspond to the warps minimizing
            'warped_distances' (since for every entity pair the one earlier on
            the list and with a smaller index has been warped to get
            'warped_distances', while the other entity remains static),
            whereas those of the lower triangular part have the opposite sign
            (due to the antisymmetric nature of pairwise warping).

        """
        warped_dist_mat = np.zeros((self.nb_var, self.nb_var))
        optimal_warp_mat = np.zeros((self.nb_var, self.nb_var))
        min_warped_distances = np.min(warped_distances, axis=0)
        warped_dist_mat[tuple(index_pairs.T)] = min_warped_distances
        warped_dist_mat[tuple(index_pairs[:, (1, 0)].T)] = min_warped_distances
        max_warp_step = int((warped_distances.shape[0] - 1)/2)
        optimal_warp = np.argmin(warped_distances, axis=0) - max_warp_step
        optimal_warp_mat[tuple(index_pairs.T)] = optimal_warp
        optimal_warp_mat[tuple(index_pairs[:, (1, 0)].T)] = -optimal_warp
        return warped_dist_mat, optimal_warp_mat.astype(int)


class Clustering(FoldChanges):
    """
    A class containing tools for clustering fold changes, inherits from
    FoldChanges class.

    Attributes
    ----------
    dist : str
        Distance chosen for clustering, 'd2hat' by default (L2 distance between
        random estimators), can also be 'wasserstein' (Wasserstein distance)
        and 'hellinger' (Hellinger distance).
    sign_pen : bool
        If True, then the distance is penalized with sign penalty.
        The default is False.
    pen_param : float
        Parameter determining the weight of sign penalty. The default is 1.
    time_warp : bool
        If True, then the clustering procedure is coupled with the alignment.
        The default is False.
    max_warp_step : int
        If max_warp_step=i>0, then the set of all considered warps is the
        set of all integers between -i and i.
    index_pairs : ndarray
        2D array of shape (number of pairs, 2) containing pairs
        of indices in the same order as the pairwise distances array
        'distances'.
    distances : ndarray
        1D array of length equal to the number of pairs containing
        pairs of distances between the fold changes in the same order as
        'index_pairs'.
    dist_mat : ndarray
        Distance matrix, 2D array of shape with both dimensions
        equal to the number of fold changes (entities). If time_warp is True,
        then the distance matrix used for clustering is OWD
        (Optimal Warping Distance) matrix containing distances that minimize
        the pairwise distance over the set of all considered warps.
    optimal_warp_mat : ndarray
        Optimal Warp matrix, 2D array with both dimensions equal to the
        number of fold changes (entities). The values of the upper
        triangular part of the matrix correspond to the warps minimizing
        'warped_distances' (since for every entity pair the one earlier on
        the list and with a smaller index has been warped to get
        'warped_distances', while the other entity remains static),
        whereas those of the lower triangular part have the opposite sign
        (due to the antisymmetric nature of pairwise warping).
        Defined if time_warp is True.
    random_gen : RandomState instance or None
        Random number generator, used to reproduce results. If None (default),
        the generator is the RandomState instance used by `np.random`.
        If RandomState instance, random_gen is the actual random
        number generator.


    Methods
    -------
    init_centroids(k)
        Initializes centroids (medoids pr barycenters) for k clusters.
    assign_clusters(centroids, method='k-medoids', wass_dist_mat=None)
        Assigns all fold changes to one of the k clusters based on their
        distances to centroids (medoids or barycenters).
    update_centroids(k, clusters, old_centroids, algorithm='k-means-like')
        Recalculates centroids based on the current cluster configuration.
    compute_barycenter(k, clusters, cov0, precision=1e-5)
        Calculates barycenters with respect to the Wasserstein distance for
        k clusters by solving a fixed point problem iteratively until the
        stopping criterion is satisfied.
    hierarchical_centroids(k, clusters)
        Chooses centroids among the fold changes in clusters after clustering.
        Used for non-centroid based clustering methods, such as hierarchical
        clustering.
    calculate_total_cost(centroids, clusters)
        Calculates total cost for all clusters, defined as the sum of distances
        between the fold changes and their centroids with respect to the
        distance matrix. Used in k-medoids clustering as a selection criterion.
    choose_k_clusters(k, method='k-medoids', algorithm='k-means-like',
                      verbose=0, plot_umap=True, nb_rep_umap=1,
                      umap_color_labels=None, plot_umap_labels=False)
        Performs clustering in k clusters of a set of random fold changes
        estimators based on one random clusters' initialization.
    fc_clustering(k, nb_rep=100, method='k-medoids', verbose=0,
                  disp_plot=False, algorithm='k-means-like', nb_best=1,
                  tree_cutoff=5, silhouette=False, umap_color_labels=None,
                  plot_umap_labels=False)
        Performs a series of clustering attempts of a set of random fold
        changes estimators for different numbers of clusters by trying
        multiple random clusters' initializations and choosing the attempt
        producing the best outcome (in the cases where random initializations
        are applicable).
    plot_clusters(k, clusters, centroids, centroid_type='medoid', warps=None,
                  nb_cols=4, nb_rows=None, figsize=None)
        Produces a figure with k subplots (or 2 figures if warps are provided),
        each containing plots of the fold changes' means in the corresponding
        cluster. In the case with time warping, produces a figure with
        unaligned (original) and a figure with aligned (with respect to their
        centroids) fold changes.

    """
    __doc__ += 'Inherited from FoldChanges: \n ' + FoldChanges.__doc__

    def __init__(self, data=None, means=None, cov=None, var_names=None,
                 time_points=None, dist='d2hat', time_warp=False,
                 max_warp_step=1, sign_pen=False, pen_param=1,
                 random_gen=None):
        """
        Parameters
        ----------
        data : ndarray or None
            If not None, 4D array with the dimensions corresponding to:
            1) nb of time points, 2) two experimental conditions
            (dim 0: control, dim 1: case)), 3) replicates, 4) nb of entities.
            If None (by default), then the fold changes are constructed based
            on 'means' and 'cov'. Either 'data' or 'means' and 'cov' have to be
            non-None, with 'data' having priority for the fold changes
            construction.
        means : ndarray or None
            If not None, 2D array of shape (nb_time_pts, nb_var)
            containing data with `float` type, representing fold changes' means
            for each entity and each time point. If 'data' is None, used to
            construct fold changes. Either 'data' or 'means' and 'cov' have to
            be non-None.
        cov : ndarray or None
            If not None, 3D array of shape (nb_time_pts, nb_var, nb_var)
            containing data with `float` type, representing fold changes'
            nb_var x nb_var shaped covariance matrices for each time point.
            Time-wise cross-covariances are assumes to be 0 due to experimental
            design. In case of Hellinger distance, can also be 4-dimensional
            (natural form): (nb_time_pts, nb_time_pts, nb_var, nb_var).
            If 'data' is None, used to construct fold changes. Either 'data' or
            'means' and 'cov' have to be non-None.
        var_names : array-like or None
            1D array-like containing data with `string` type, representing
            names of the measured entities (ex. genes). The default is None.
        time_points : array-like or None
            1D array-like containing data with `float` type, representing time
            points at which fold changes were measured. The default is None.
        dist : str
            Distance chosen for clustering, 'd2hat' by default (L2 distance
            between random estimators), can also be 'wasserstein'
            (Wasserstein distance) and 'hellinger' (Hellinger distance).
        time_warp : bool
            If True, then the clustering procedure is coupled with the
            alignment. The default is False.
        max_warp_step : int
            If max_warp_step=i>0, then the set of all considered warps is the
            set of all integers between -i and i.
        sign_pen : bool
            If True, then the distance is penalized with sign penalty.
            The default is False.
        pen_param : float
            Parameter determining the weight of sign penalty. The default is 1.
        random_gen : RandomState instance or None
            Random number generator, used to reproduce results. If None
            (default), the generator is the RandomState instance used by
            `np.random`. If RandomState instance, random_gen is the actual
            random number generator.

        """
        super().__init__(data=data, means=means, cov=cov,
                         var_names=var_names, time_points=time_points)
        self.dist = dist
        self.time_warp = time_warp
        if sign_pen:
            self.sign_pen = sign_pen
            self.pen_param = pen_param
        # With alignment:
        if time_warp:
            assert max_warp_step >= 0
            self.max_warp_step = max_warp_step
            if self.means is not None:
                (self.index_pairs,
                 warped_distances) = (self.compute_warped_distance_pairs(max_warp_step=max_warp_step,
                                                                         sign_pen=sign_pen,
                                                                         pen_param=pen_param))
                self.distances = np.min(warped_distances, axis=0)
        # Without alignment:
        else:
            if self.means is not None:
                (self.index_pairs,
                 self.distances) = self.compute_distance_pairs(dist=self.dist,
                                                               sign_pen=sign_pen,
                                                               pen_param=pen_param)
        if self.means is not None:
            if self.dist in ('d2hat', 'hellinger'):
                if time_warp:
                    (self.dist_mat,
                     self.optimal_warp_mat) = self.compute_warped_dist_mat(self.index_pairs,
                                                                           warped_distances)
                else:
                    self.dist_mat = self.compute_dist_mat(self.index_pairs,
                                                          self.distances)
            if self.dist == 'wasserstein':
                fc_var = np.diagonal(self.cov, axis1=1, axis2=2)
                id_tensor = (np.repeat(np.identity(self.nb_time_pts),
                                       self.nb_var, axis=1)
                             .reshape((self.nb_time_pts, self.nb_time_pts,
                                       self.nb_var)))
                M1 = np.einsum('ijk,ik->ijk', id_tensor, fc_var)
                self.dist_mat = self.compute_cross_distances(self.means, M1)[0]
        if self.means is None:
            self.dist_mat = None
        if random_gen:
            self.random_gen = random_gen
        else:
            self.random_gen = np.random

    def init_centroids(self, k):
        """
        Produces a set of k random centroids to initialize clustering according
        to the algorithm k-means++.

        Parameters
        ----------
        k : int
            Number of clusters.

        Returns
        -------
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that have been chosen as initial centroids.

        """
        # Choose the first centroid randomly:
        centroid = self.random_gen.randint(0, self.nb_var, 1)[0]
        centroids = np.zeros((k))
        centroids[0] = centroid
        dist_array = np.zeros((k, self.nb_var))
        # Choose the remaining k-1 centroids with probabilities that tend to
        # maximize the distances between them:
        for i in range(k - 1):
            all_vs_centroid = np.argwhere((self.index_pairs[:, 0] == centroid)
                                          | (self.index_pairs[:, 1] == centroid))
            dist_array[i, :centroid] = np.squeeze(
                self.distances[all_vs_centroid[:centroid]])
            dist_array[i, (centroid+1)                       :] = np.squeeze(self.distances[all_vs_centroid[centroid:]])
            min_dist = np.min(dist_array[:(i+1), :], axis=0)
            # A fold change is chosen as the next centroid with a probability
            # proportional to its distance to the closest centroid among those
            # that have already been chosen:
            prob = min_dist/np.sum(min_dist)
            centroid = self.random_gen.choice(range(self.nb_var), 1, p=prob)[0]
            centroids[i+1] = centroid
        return centroids

    def assign_clusters(self, centroids, method='k-medoids', wass_dist_mat=None):
        """
        Assigns all fold changes to one of the k clusters based on their
        distances to centroids (medoids or barycenters).

        Parameters
        ----------
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as current centroids (medoids). Used for
            clusters assignment only if method=='k-medoids'.
        method : str, optional
            Main approach to clustering, either 'k-medoids' (default, coupled
            with d2hat distance or Hellinger distance) or 'wass k-means'
            (Wasserstein k-means).
        wass_dist_mat : ndarray, optional
            2D array of shape (k, nb_var) containing distances between the
            fold changes and the barycenters for all clusters. Used for
            clusters assignment only if method=='wass k-means', otherwise
            None (by default).
        Returns
        -------
        clusters : ndarray
            1D array of length nb_var containing integers in range (0, k)
            indicating clusters to which the fold changes are assigned.

        """
        centroids_int = centroids.astype(int)
        if method == 'k-medoids':
            all_vs_centroids = np.copy(self.dist_mat[centroids_int, :])
            clusters = np.argmin(all_vs_centroids, axis=0)
        if method == 'wass k-means':
            clusters = np.argmin(wass_dist_mat, axis=0)
            nb_cl = wass_dist_mat.shape[0]
            if len(np.unique(clusters)) < nb_cl:
                clusters_to_fill = np.setdiff1d(range(nb_cl), clusters)
                ind = self.random_gen.randint(0, len(clusters),
                                              len(clusters_to_fill))
                clusters[ind] = clusters_to_fill
        return clusters

    def update_centroids(self, k, clusters, old_centroids,
                         algorithm='k-means-like'):
        """
        Recalculates centroids based on the current cluster configuration.

        Parameters
        ----------
        k : int
            Number of clusters.
        clusters : ndarray
            1D array of length nb_var containing integers in range (0, k)
            indicating clusters to which the fold changes are assigned.
        old_centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as current centroids (medoids)
            before the update.
        algorithm : str, optional
            Indicates a choice of one of the two common variations of k-medoids
            clustering. The default is 'k-means-like' (Park, 2006), can also
            be 'PAM' (Partitioning Around Medoids; Schubert, Rousseeuw, 2019).

        Returns
        -------
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as current centroids (medoids)
            after the update.

        """
        centroids_int = old_centroids.astype(int)
        new_centroids = np.zeros((k), dtype=int)
        new_dist = np.zeros((k))
        # k-means-like, or original k-medoids:
        if algorithm == 'k-means-like':
            for i in range(k):
                cluster_i = np.squeeze(np.argwhere(clusters == i), axis=1)
                total_clust_dist = np.sum(self.dist_mat[cluster_i, :][:, cluster_i],
                                          axis=0)
                new_centroids[i] = cluster_i[np.argmin(total_clust_dist)]
                new_dist[i] = np.min(total_clust_dist)
            centroids = new_centroids.copy()
        # Partitioning Around Medoids (PAM) algorithm
        if algorithm == 'PAM':
            for i in range(k):
                cluster_i = np.squeeze(np.argwhere(clusters == i), axis=1)
                # delta_dist corresponds to the cost of changing current
                # medoids:
                delta_dist = (np.sum(self.dist_mat[cluster_i, :], axis=0)
                              - self.dist_mat[np.arange(self.nb_var),
                                              clusters.astype(int)])
                # Force to change medoids to get an additional chance to
                # minimize the criterion:
                delta_dist[centroids_int] = np.inf
                new_centroids[i] = np.argmin(delta_dist)
                new_dist[i] = np.min(delta_dist)
            centroids = centroids_int.copy()
            centroids[np.argmin(new_dist)] = new_centroids[np.argmin(new_dist)]
        return centroids

    @staticmethod
    def _minus_sqrt_mat(m):
        """
        Computes a -0.5 power of a matrix by diagonalization, used in
        'compute_barycenter' to calculate one of the terms of the fixed point
        equation in order to find an approximation of the Wasserstein
        barycenter's covariance matrix.

        Parameters
        ----------
        m : ndarray
            2D array, in 'compute_barycenter': barycenter covariance matrix of
            shape (nb_time_pts, nb_time_pts).

        Returns
        -------
        ndarray
            -0.5 power of m.

        """
        eig_val, eig_vec = np.linalg.eigh(m)
        return eig_vec @ np.diag(np.power(eig_val, -0.5)) @ (eig_vec.T)

    def compute_barycenter(self, k, clusters, cov0, precision=1e-5):
        """
        Calculates barycenters with respect to the Wasserstein distance for
        k clusters by solving a fixed point problem iteratively until the
        stopping criterion is satisfied.

        Parameters
        ----------
        k : int
            Number of clusters.
        clusters : ndarray
            1D array of length equal to 'nb_var' with values of type 'int'
            between 0 and k-1 indicating which cluster every fold change
            belongs to.
        cov0 : ndarray
            2D array of shape (nb_time_pts, nb_time_pts), a symmetric positive
            definite matrix that initializes the barycenters' covariance
            matrices.
        precision : float, optional
            Stopping criterion, the fixed point equation iterations stop when
            the difference between the old and the new total costs for the
            considered cluster becomes smaller or equal to this value.
            The default is 1e-5.

        Returns
        -------
        bary_means : ndarray
            2D array of shape (nb_time_pts, k) representing final barycenter
            means for all clusters.
        bary_cov : ndarray
            3D array of shape (nb_time_pts, nb_time_pts, k) representing
            final barycenter covariance matrices for all clusters.
        all_costs : ndarray
            1D array of length k containing final total costs per cluster.

        """
        bary_means = np.zeros((self.nb_time_pts, k))
        bary_cov = np.zeros((self.nb_time_pts, self.nb_time_pts, k))
        all_costs = np.zeros((k))
        for i in range(k):
            cluster_i = np.squeeze(np.argwhere(clusters == i), axis=1)
            # Barycenter means are calculated with a simple average:
            bary_means[:, i] = np.mean(self.means[:, cluster_i], axis=1)
            bary_m = np.reshape(bary_means[:, i], (self.nb_time_pts, 1))
            bary_c = np.copy(cov0)
            dist_cluster_i = np.zeros((cluster_i.size))
            it = 0
            delta_cost = np.inf
            old_cost = np.inf

            # Fixed point equation iteration to find the optimal barycenter
            # covariance matrix (approximation):
            while delta_cost > precision:
                #print('iter ', it)
                if it > 100:
                    break
                bary_c = np.reshape(bary_c, (self.nb_time_pts,
                                             self.nb_time_pts, 1))
                (dist_cluster_i,
                 K) = self.compute_cross_distances(bary_m,
                                                   bary_c,
                                                   cluster=cluster_i)
                K_mean = np.mean(K, axis=(0, 1))
                new_cost = np.sum(dist_cluster_i)
                delta_cost = np.abs(old_cost - new_cost)
                old_cost = new_cost
                bary_c_power = Clustering._minus_sqrt_mat(np.squeeze(bary_c))
                try:
                    bary_c = (bary_c_power @ np.linalg.matrix_power(K_mean, 2)
                              @ bary_c_power)
                except np.linalg.LinAlgError:
                    break
                it += 1
            bary_cov[:, :, i] = np.copy(bary_c)
            all_costs[i] = new_cost
        return bary_means, bary_cov, all_costs

    def hierarchical_centroids(self, k, clusters):
        """
        Chooses centroids among the fold changes in clusters after clustering.
        Used for non-centroid based clustering methods, such as hierarchical
        clustering.

        Parameters
        ----------
        k : int
            Number of clusters.
        clusters : ndarray
            1D array of length equal to 'nb_var' with values of type 'int'
            between 0 and k-1 indicating which cluster every fold change
            belongs to.

        Returns
        -------
        ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that represent cluster centroids.

        """
        centroids = np.zeros((k), dtype=int)
        for i in range(k):
            cluster_i = np.squeeze(np.argwhere(clusters == i), axis=1)
            dist_mat_cl_i = self.dist_mat[cluster_i, :][:, cluster_i]
            total_clust_dist = np.squeeze(np.sum(dist_mat_cl_i, axis=0,
                                                 keepdims=True))
            try:
                centroid_i = np.argmin(total_clust_dist)
            # Error for example if a cluster is empty
            except ValueError:
                return np.repeat(np.nan, k)
            centroids[i] = cluster_i[centroid_i]
        return centroids

    @staticmethod
    def _plot_dendrogram(model, **kwargs):
        """
        Code from: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

        Creates linkage matrix and then plots the dendrogram. Used in
        hierarchical clustering to choose the number of clusters.

        """

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    def calculate_total_cost(self, centroids, clusters):
        """
        Calculates total cost for all clusters, defined as the sum of distances
        between the fold changes and their centroids with respect to the
        distance matrix. Used in k-medoids clustering as a selection criterion.

        Parameters
        ----------
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as current centroids.
        clusters : ndarray
            1D array of length nb_var containing integers in range (0, k)
            indicating clusters to which the fold changes are assigned.

        Returns
        -------
        float
            Value of the total cost.

        """
        costs = np.zeros((len(centroids)))
        centroids_int = centroids.astype(int)
        for i, c in enumerate(centroids_int):
            cluster_i = np.squeeze(np.argwhere(clusters == i), axis=1)
            costs[i] = np.sum(self.dist_mat[c, cluster_i])
        return np.sum(costs)

    def calculate_comparable_cost(self, k, clusters):
        """
        Calculates total comparable cost for all clusters, defined as the sum
        of distances between all fold change pairs in each cluster with respect
        to the distance matrix. Used to compare clustering performed with
        different methods (distance matrix should be the same).

        Parameters
        ----------
        k : int
            Number of clusters.
        clusters : ndarray
            1D array of length nb_var containing integers in range (0, k)
            indicating clusters to which the fold changes are assigned.

        Returns
        -------
        float
            Value of the total comparable cost.

        """
        costs = np.zeros((k))
        for i in range(k):
            cluster_i = np.squeeze(np.argwhere(clusters == i), axis=1)
            cluster_dist = self.dist_mat[tuple(
                cluster_i), :][:, tuple(cluster_i)]
            costs[i] = np.sum(cluster_dist)/2
        return np.sum(costs)

    def choose_k_clusters(self, k, method='k-medoids', algorithm='k-means-like',
                          verbose=0, plot_umap=True, nb_rep_umap=1,
                          umap_color_labels=None, plot_umap_labels=False):
        """
        Performs clustering in k clusters of a set of random fold changes
        estimators based on one random clusters' initialization.

        Parameters
        ----------
        k : int
            Number of clusters.
        method : str, optional
            Main approach to clustering, options include:
                - 'k-medoids' (default, coupled with d2hat distance or
                Hellinger distance),
                - 'wass k-means' (Wasserstein k-means),
                - 'hierarchical' (hierarchical clustering based on
                d2hat distance),
                - 'umap' (UMAP projection of the d2hat distance matrix with
                subsequent k-means clustering of the projection coordinates).
        algorithm : str, optional
            Indicates a choice of one of the two common variations of k-medoids
            clustering. The default is 'k-means-like' (Park, 2006), can also
            be 'PAM' (Partitioning Around Medoids; Schubert, Rousseeuw, 2019).
        verbose : int, optional
            Controls the verbosity, if 1 (or larger) then informs on the
            advancement of clustering.
        plot_umap : bool, optional
            If True (default) and method is 'umap', then plots the UMAP
            projection of the distance matrix.
        nb_rep_umap : int, optional
            Number of k-means clustering initializations performed on the
            UMAP projection, relevant if method is 'umap'. The default is 1.
        umap_color_labels : None or array-like, optional
            Relevant if method is 'umap'. If None (default), then the data
            points on the UMAP projection are colored with respect to the
            cluster labels assigned by k-means. Alternatively, can be a 1D
            array-like of length equal to nb_var, containing integers
            indicating cluster labels assigned to the fold changes.  In this
            case, colors are chosen corresponding to these labels. This option
            is intended for use in the framework of simulation studies.
        plot_umap_labels : bool, optional
            Relevant if method is 'umap'. If True, then labels the data points
            on the UMAP projection with corresponding fold changes' indices.
            The default is False (no labels).

        Returns
        -------
        List containing the following elements:
            clusters : ndarray
                1D array of length nb_var containing integers in range (0, k)
                indicating clusters to which the fold changes are assigned.
                Returned as the first element of the list in all cases.
            centroids : ndarray
                1D array of length k containing indices in range (0, nb_var) of
                the fold changes that have been chosen as centroids.
                Returned as the second element of the list in all
                cases except if method=='wass k-means'.
            bary_means : ndarray
                2D array of shape (nb_time_pts, k) representing final
                barycenter means for all clusters. Returned as the second
                element of the list if method=='wass k-means'.
            bary_cov : ndarray
                3D array of shape (nb_time_pts, nb_time_pts, k) representing
                final barycenter covariance matrices for all clusters. Returned
                as the third element of the list if method=='wass k-means'.
            total_cost : float
                Value of the final total clustering cost with respect to the
                metric associated with the chosen clustering method.
                Returned as the last element of the list if method=='k-medoids'
                or method=='wass k-means' (in other cases absent since the cost
                isn't assessed during clustering and should be calculated
                separately if needed).

        """
        # Initialize clusters with k-means++ (used for d2hat k-medoids and
        # Wasserstein k-means):
        centroids = self.init_centroids(k)

        # Iterate assign clusters & recalculate centroids until
        # criterion is satisfied :
        flag = False
        i = 0
        if method == 'k-medoids':
            while not flag:
                clusters = self.assign_clusters(centroids, method=method)
                total_cost = self.calculate_total_cost(centroids, clusters)
                if verbose > 0:
                    print('Iteration ', i)
                    print('Total cost: ', total_cost)
                new_centroids = self.update_centroids(k, clusters,
                                                      centroids,
                                                      algorithm=algorithm)
                new_cost = self.calculate_total_cost(new_centroids, clusters)
                flag = new_cost >= total_cost
                if not flag and (new_cost >= 0):
                    centroids = np.copy(new_centroids)
                    total_cost = new_cost
                i += 1
            return clusters, centroids, total_cost
        if method == 'wass k-means':
            total_cost = np.inf
            centroids_int = centroids.astype(int)
            bary_means = self.means[:, centroids_int]
            bary_var = self.cov[:, centroids_int, centroids_int]
            id_tensor = (np.repeat(np.identity(self.nb_time_pts), k, axis=1).
                         reshape((self.nb_time_pts, self.nb_time_pts, k)))
            bary_cov = np.einsum('ijk,ik->ijk', id_tensor, bary_var)
            while not flag:
                cov0 = spd(self.nb_time_pts)
                wass_dist_mat = self.compute_cross_distances(bary_means,
                                                             bary_cov)[0]
                clusters = self.assign_clusters(centroids, method=method,
                                                wass_dist_mat=wass_dist_mat)
                (new_bary_means, new_bary_cov,
                 new_costs) = self.compute_barycenter(k, clusters, cov0)
                new_total_cost = np.sum(new_costs)
                flag = new_total_cost >= total_cost
                if not flag and (new_total_cost >= 0):
                    bary_means = np.copy(new_bary_means)
                    bary_cov = np.copy(new_bary_cov)
                    total_cost = new_total_cost
                if verbose > 0:
                    print('Iteration ', i)
                    print('Total cost: ', total_cost)
                i += 1
            return clusters, bary_means, bary_cov, total_cost
        if method == 'hierarchical':
            h_clustering = AgglomerativeClustering(n_clusters=k, affinity='precomputed',
                                                   linkage='complete')
            clusters = h_clustering.fit_predict(self.dist_mat)
            centroids = self.hierarchical_centroids(k, clusters)
            return clusters, centroids
        if method == 'umap':
            warnings.filterwarnings('ignore', '.*precomputed metric.*')
            fit_umap = umap.UMAP(metric='precomputed',
                                 random_state=self.random_gen).fit_transform(self.dist_mat)
            sp_cl = KMeans(n_clusters=k, n_init=nb_rep_umap,
                           random_state=self.random_gen).fit(fit_umap)
            clusters = sp_cl.labels_
            centroids = self.hierarchical_centroids(k, clusters)

            if plot_umap:
                if umap_color_labels is None:
                    cmap = sns.color_palette(n_colors=clusters.max()+1)
                    colors = np.array(cmap)[clusters]
                else:
                    cmap = sns.color_palette(
                        n_colors=umap_color_labels.max()+1)
                    colors = np.array(cmap)[umap_color_labels]
                legend_elements = [Line2D([0], [0], marker='o', color=c,
                                          markersize=7, linewidth=0,
                                          label='Cluster '+str(i+1)) for i, c in enumerate(cmap)]
                fig = plt.figure(figsize=(12, 5))
                ax = fig.subplots(1, 1)
                mpl.rcParams['font.family'] = 'serif'
                ax.scatter(fit_umap[:, 0], fit_umap[:, 1], c=colors, s=70,
                           alpha=0.7)
                ax.legend(handles=legend_elements, fontsize=12)
                if plot_umap_labels:
                    for i, txt in enumerate(range(self.nb_var)):
                        ax.annotate(txt, (fit_umap[i, 0], fit_umap[i, 1]),
                                    fontsize=11)
                plt.title(f'UMAP, {k} clusters', fontsize=15)
                plt.show()
            return clusters, centroids

    def fc_clustering(self, k, nb_rep=100, method='k-medoids', verbose=0,
                      disp_plot=False, algorithm='k-means-like', nb_best=1,
                      tree_cutoff=5, silhouette=False, umap_color_labels=None,
                      plot_umap_labels=False):
        """
        Performs a series of clustering attempts of a set of random fold
        changes estimators for different numbers of clusters by trying
        multiple random clusters' initializations and choosing the attempt
        producing the best outcome (in the cases where random initializations
        are applicable).

        Parameters
        ----------
        k : int
            Number of clusters.
        nb_rep : int, optional
            Number of random initialization attempts (k-means clustering
            initializations performed on the UMAP projection if
            method=='umap'). The default is 100.
        method : str, optional
            Main approach to clustering, options include:
                - 'k-medoids' (default, coupled with d2hat distance or
                Hellinger distance),
                - 'wass k-means' (Wasserstein k-means),
                - 'hierarchical' (hierarchical clustering based on
                d2hat distance),
                - 'umap' (UMAP projection of the d2hat distance matrix with
                subsequent k-means clustering of the projection coordinates).
        algorithm : str, optional
            Indicates a choice of one of the two common variations of k-medoids
            clustering. The default is 'k-means-like' (Park, 2006), can also
            be 'PAM' (Partitioning Around Medoids; Schubert, Rousseeuw, 2019).
        verbose : int, optional
            Controls the verbosity, if 1 (or larger) then informs on the
            advancement of clustering.
        disp_plot : bool, optional
            False by default, if True then plots the mean total cost curve with
            standard deviations or the UMAP projection of the distance matrix
            depending on the method.
        nb_best : int, optional
            Number of the best random initialization attempts to be taken into
            account for the total cost plot. The default is 1.
        tree_cutoff : int, optional
            Relevant if method is 'hierarchical', the number of dendrogram tree
            levels that are displayed. The default if 5.
        silhouette : bool, optional
            The default is False, if True then the mean silhouette score curve
            with standard deviations is displayed along with the total costs.
        umap_color_labels : None or array-like, optional
            Relevant if method is 'umap'. If None (default), then the data
            points on the UMAP projection are colored with respect to the
            cluster labels assigned by k-means. Alternatively, can be a 1D
            array-like of length equal to nb_var, containing integers
            indicating cluster labels assigned to the fold changes.  In this
            case, colors are chosen corresponding to these labels. This option
            is intended for use in the framework of simulation studies.
        plot_umap_labels : bool, optional
            Relevant if method is 'umap'. If True, then labels the data points
            on the UMAP projection with corresponding fold changes' indices.
            The default is False (no labels).

        Returns
        -------
        If k is an integer, returns same as choose_k_clusters. If k is a
        container with integers, then returns a list of dictionaries, with
        keys corresponding to the considered numbers of clusters, and the
        values are the same as returned by choose_k_clusters.
        If time_warp is True, an new element warps (or all_warps if
        if different numbers of clusters are considered) is added to the list
        for all distance matrix-based methods (i.e. all except 'wass k-means').
        For a fixed number of clusters it is a 1D array of length nb_var
        containing integers in range (-max_warp_step, max_warp_step + 1)
        indicating fold changes' warps with respect to their
        corresponding centroids.

        """
        if method == 'k-medoids':
            if np.size(k) == 1:
                total_cost = np.inf
                for i in range(nb_rep):
                    (clusters_i, centroids_i,
                     total_cost_i) = self.choose_k_clusters(k, method=method,
                                                            verbose=verbose,
                                                            algorithm=algorithm)
                    if total_cost_i < total_cost:
                        clusters = np.copy(clusters_i)
                        centroids = np.copy(centroids_i)
                        total_cost = total_cost_i
                if self.time_warp:
                    warps = self.optimal_warp_mat[np.arange(0, self.nb_var),
                                                  centroids[clusters]]
                    return clusters, centroids, warps, total_cost
                return clusters, centroids, total_cost
            else:
                costs_array = np.zeros((len(k)))
                silhouette_score_array = np.zeros((len(k)))
                all_costs = {}
                all_clusters = {}
                all_centroids = {}
                if self.time_warp:
                    all_warps = {}
                if nb_best > 1:
                    mean_best_costs = np.zeros((len(k)))
                    std_best_costs = np.zeros((len(k)))
                    mean_best_silhouette = np.zeros((len(k)))
                    std_best_silhouette = np.zeros((len(k)))
                for it, j in enumerate(k):
                    if verbose > 0:
                        print('Cluster ', j+1)
                    total_cost = np.inf
                    if nb_best > 1:
                        best_costs = np.ones((nb_best))*np.inf
                        best_silhouette = np.ones((nb_best))*np.inf
                    for i in range(nb_rep):
                        (clusters_i, centroids_i,
                         total_cost_i) = self.choose_k_clusters(j, method=method,
                                                                verbose=verbose,
                                                                algorithm=algorithm)
                        if total_cost_i < total_cost:
                            clusters = np.copy(clusters_i)
                            centroids = np.copy(centroids_i)
                            total_cost = total_cost_i
                        if nb_best > 1:
                            if total_cost_i < max(best_costs):
                                best_costs[np.argmax(
                                    best_costs)] = total_cost_i
                                if silhouette:
                                    best_silhouette[np.argmax(best_costs)] = silhouette_score(self.dist_mat,
                                                                                              clusters_i,
                                                                                              metric="precomputed")
                    costs_array[it] = total_cost
                    if silhouette:
                        silhouette_score_array[it] = silhouette_score(self.dist_mat,
                                                                      clusters,
                                                                      metric="precomputed")
                    all_costs[str(j)] = total_cost
                    all_clusters[str(j)] = clusters
                    all_centroids[str(j)] = centroids
                    if nb_best > 1:
                        mean_best_costs[it] = np.mean(best_costs)
                        std_best_costs[it] = np.std(best_costs)
                        mean_best_silhouette[it] = np.mean(best_silhouette)
                        std_best_silhouette[it] = np.std(best_silhouette)
                    if self.time_warp:
                        corresp_centr = all_centroids[str(j)][all_clusters[str(j)]]
                        all_warps[str(j)] = self.optimal_warp_mat[np.arange(0, self.nb_var),
                                                                  corresp_centr]
                if disp_plot:
                    if silhouette:
                        fig = plt.figure(figsize=(12, 5))
                        axs = fig.subplots(1, 2)
                        axs[1].set_ylim(0, max(mean_best_silhouette)+0.2)
                        if nb_best <= 1:
                            axs[0].title.set_text(
                                'Total clustering costs (best result)')
                            axs[0].plot(k, costs_array, 'b')
                            axs[0].xlabel('Num. of clusters')
                            axs[1].plot(k, silhouette_score_array, 'g')
                            axs[1].title.set_text(
                                'Silhouette score (best result)')
                            axs[1].set_xlabel('Num. of clusters')
                        else:
                            axs[0].title.set_text(
                                f'Total clustering costs (mean  & std of {nb_best/nb_rep*100}% best results)')
                            axs[0].plot(k, mean_best_costs, 'b')
                            axs[0].set_xlabel('Num. of clusters')
                            axs[0].errorbar(k, mean_best_costs, std_best_costs,
                                            fmt='None', linestyle='', ecolor='grey',
                                            capsize=2.5, capthick=2, alpha=0.4)
                            axs[1].plot(k, mean_best_silhouette, 'g')
                            axs[1].set_xlabel('Num. of clusters')
                            axs[1].errorbar(k, mean_best_silhouette, std_best_silhouette,
                                            fmt='None', linestyle='', ecolor='grey',
                                            capsize=2.5, capthick=2, alpha=0.4)
                            print(mean_best_silhouette)
                            axs[1].title.set_text(
                                f'Silhouette score (mean  & std of {nb_best/nb_rep*100}% best results)')
                    else:
                        fig = plt.figure(figsize=(12, 5))
                        if nb_best <= 1:
                            plt.plot(k, costs_array, 'b')
                            plt.title('Total clustering costs (best result)')
                            plt.xlabel('Num. of clusters')
                        else:
                            plt.plot(k, mean_best_costs, 'b')
                            plt.xlabel('Num. of clusters')
                            plt.errorbar(k, mean_best_costs, std_best_costs,
                                         fmt='None', linestyle='', ecolor='grey',
                                         capsize=2.5, capthick=2, alpha=0.4)
                            plt.title(
                                f'Total clustering costs (mean  & std of {nb_best/nb_rep*100}% best results)')
                    plt.show()
                if self.time_warp:
                    return all_clusters, all_centroids, all_warps, all_costs
                all_clusters, all_centroids, all_costs
        if method == 'wass k-means':
            if np.size(k) == 1:
                total_cost = np.inf
                for i in range(nb_rep):
                    if i % 50 == 0 and verbose > 0:
                        print('rep ', i)
                    (clusters_i,
                     bary_means_i,
                     bary_cov_i,
                     total_cost_i) = self.choose_k_clusters(k, method=method,
                                                            verbose=verbose)
                    if round(total_cost_i, 6) < round(total_cost, 6):
                        clusters = np.copy(clusters_i)
                        bary_means = np.copy(bary_means_i)
                        bary_cov = np.copy(bary_cov_i)
                        total_cost = total_cost_i
                return clusters, bary_means, bary_cov, total_cost
            else:
                costs_array = np.zeros((len(k)))
                sil_score_array = np.zeros((len(k)))
                all_costs = {}
                all_clusters = {}
                all_bary_means = {}
                all_bary_cov = {}
                if nb_best > 1:
                    mean_best_costs = np.zeros((len(k)))
                    std_best_costs = np.zeros((len(k)))
                    mean_best_silhouette = np.zeros((len(k)))
                    std_best_silhouette = np.zeros((len(k)))
                for it, j in enumerate(k):
                    if verbose > 0:
                        print('Cluster ', j+1)
                    total_cost = np.inf
                    if nb_best > 1:
                        best_costs = np.ones((nb_best))*np.inf
                        best_silhouette = np.ones((nb_best))*np.inf
                    for i in range(nb_rep):
                        (clusters_i,
                         bary_means_i,
                         bary_cov_i,
                         total_cost_i) = self.choose_k_clusters(j, method=method,
                                                                verbose=verbose)
                        if total_cost_i < total_cost:
                            clusters = np.copy(clusters_i)
                            bary_means = np.copy(bary_means_i)
                            bary_cov = np.copy(bary_cov_i)
                            total_cost = total_cost_i
                        if nb_best > 1:
                            if total_cost_i < max(best_costs):
                                best_costs[np.argmax(
                                    best_costs)] = total_cost_i
                                if silhouette:
                                    best_silhouette[np.argmax(best_costs)] = silhouette_score(self.dist_mat,
                                                                                              clusters_i,
                                                                                              metric="precomputed")
                    costs_array[it] = total_cost
                    if silhouette:
                        sil_score_array[it] = silhouette_score(self.dist_mat,
                                                               clusters,
                                                               metric="precomputed")
                    all_costs[str(j)] = total_cost
                    all_clusters[str(j)] = clusters
                    all_bary_means[str(j)] = bary_means
                    all_bary_cov[str(j)] = bary_cov
                    if nb_best > 1:
                        mean_best_costs[it] = np.mean(best_costs)
                        std_best_costs[it] = np.std(best_costs)
                if disp_plot is True:
                    if silhouette:
                        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                        if nb_best <= 1:
                            axs[0].plot(k, costs_array, 'b')
                            axs[0].title.set_text(
                                'Total clustering costs (best result)')
                            axs[0].xlabel('Num. of clusters')
                            axs[1].plot(k, silhouette_score_array, 'g')
                            axs[1].title.set_text(
                                'Silhouette score (best result)')
                            axs[1].xlabel('Num. of clusters')
                        else:
                            axs[0].plot(k, mean_best_costs, 'b')
                            axs[0].xlabel('Num. of clusters')
                            axs[0].errorbar(k, mean_best_costs, std_best_costs,
                                            fmt='None', linestyle='', ecolor='grey',
                                            capsize=2.5, capthick=2, alpha=0.4)
                            axs[0].title.set_text(
                                f'Total clustering costs (mean  & std of {nb_best/nb_rep*100}% best results)')
                            axs[1].plot(k, mean_best_silhouette, 'g')
                            axs[1].xlabel('Num. of clusters')
                            axs[1].errorbar(k, mean_best_silhouette, std_best_silhouette,
                                            fmt='None', linestyle='', ecolor='grey',
                                            capsize=2.5, capthick=2, alpha=0.4)
                            print(mean_best_silhouette)
                            axs[1].title.set_text(
                                f'Silhouette score (mean  & std of {nb_best/nb_rep*100}% best results)')
                    else:
                        fig = plt.figure(figsize=(12, 5))
                        if nb_best <= 1:
                            plt.plot(k, costs_array, 'b')
                            plt.title('Total clustering costs (best result)')
                            plt.xlabel('Num. of clusters')
                        else:
                            plt.plot(k, mean_best_costs, 'b')
                            plt.xlabel('Num. of clusters')
                            plt.errorbar(k, mean_best_costs, std_best_costs,
                                         fmt='None', linestyle='', ecolor='grey',
                                         capsize=2.5, capthick=2, alpha=0.4)
                            plt.title(
                                f'Total clustering costs (mean  & std of {nb_best/nb_rep*100}% best results)')
                    plt.show()
                return all_clusters, all_bary_means, all_bary_cov, all_costs
        if method == 'hierarchical' or method == 'umap':
            if np.size(k) == 1:
                (clusters,
                 centroids) = self.choose_k_clusters(k, method=method,
                                                     plot_umap=disp_plot,
                                                     nb_rep_umap=nb_rep,
                                                     umap_color_labels=umap_color_labels,
                                                     plot_umap_labels=plot_umap_labels)
                if self.time_warp:
                    warps = self.optimal_warp_mat[np.arange(0, self.nb_var),
                                                  centroids[clusters]]
                    return clusters, centroids, warps
                return clusters, centroids
            else:
                all_clusters = {}
                all_centroids = {}
                if self.time_warp:
                    all_warps = {}
                for it, j in enumerate(k):
                    if verbose > 0:
                        print('Cluster ', j)
                    (clusters,
                     centroids) = self.choose_k_clusters(j, method=method,
                                                         plot_umap=disp_plot,
                                                         nb_rep_umap=nb_rep)
                    all_clusters[str(j)] = clusters
                    all_centroids[str(j)] = centroids
                    if self.time_warp:
                        corresp_centr = all_centroids[str(
                            j)][all_clusters[str(j)]]
                        all_warps[str(j)] = self.optimal_warp_mat[np.arange(0, self.nb_var),
                                                                  corresp_centr]
                if method == 'hierarchical' and disp_plot is True:
                    model = AgglomerativeClustering(distance_threshold=0,
                                                    n_clusters=None,
                                                    affinity='precomputed',
                                                    linkage='complete')
                    model = model.fit(self.dist_mat)
                    plt.figure(figsize=(10, 3))
                    plt.title('Hierarchical Clustering Dendrogram')
                    # plot the top p levels of the dendrogram
                    Clustering._plot_dendrogram(model, truncate_mode='level',
                                                p=tree_cutoff)
                    plt.xlabel(
                        "Number of points in node (or index of point if no parenthesis).")
                    plt.show()
                if self.time_warp:
                    return all_clusters, all_centroids, all_warps
                return all_clusters, all_centroids

    def plot_clusters(self, k, clusters, centroids, centroid_type='medoid',
                      warps=None, nb_cols=4, nb_rows=None, figsize=None):
        """
        Produces a figure with k subplots (or 2 figures if warps are provided),
        each containing plots of the fold changes' means in the corresponding
        cluster. In the case with time warping, produces a figure with
        unaligned (original) and a figure with aligned (with respect to their
        centroids) fold changes.

        Parameters
        ----------
        k : int
            Number of clusters.
        clusters : ndarray or dictionary
            If ndarray, 1D array of length nb_var containing integers in range
            (0, k) indicating clusters to which the fold changes are assigned.
            If a dictionary, the keys are numbers of clusters considered, and
            for each such number the value is the latter array.
        centroids : ndarray or dictionary
            If centroid_type=='medoid':
                If ndarray, 1D array of length k containing indices in range
                (0, nb_var) of the fold changes that act as centroids.
                If a dictionary, the keys are numbers of clusters considered,
                and for each such number the value is the latter array.
            If centroid_type=='barycenter':
                If ndarray, an array of barycenter means: 2D array of shape
                (nb_time_pts, k) representing final barycenter means for all
                clusters. If a dictionary, the keys are numbers of clusters
                considered, and for each such number the value is the latter
                array.
        centroid_type : str, optional
            The default is 'medoid', in which case the centroids are selected
            among the fold changes (see centroids). Another option is
            'barycenter', in this case the barycenters are plotted based on
            their means.
        warps : ndarray or dictionary, optional
            If ndarray, 1D array of length nb_var containing integers in range
            (-max_warp_step, max_warp_step + 1) indicating fold changes' warps
            with respect to their corresponding centroids. If a dictionary,
            the keys are numbers of clusters considered, and for each such
            number the value is the latter array. The default is None,
            otherwise the versions with and without time warping are plotted.
        nb_cols : int, optional
            Number of columns of the subplot grid. The default is 4.
        nb_rows : TYPE, optional
            Number of rows of the subplot grid The default is None, in which
            case nb_rows=int(np.ceil(k/nb_cols)).
        figsize : (float, float), optional
            Width and height of the figure(s). The default is None, in which
            case figsize=(15, 6*nb_rows).

        Returns
        -------
        None.

        """
        sns.set_style("darkgrid")
        # If dictionaries are given, extract the elements corresponding to
        # the clustering with k clusters:
        if isinstance(clusters, dict):
            clusters_k = clusters[str(k)]
            centroids_k = centroids[str(k)]
        else:
            clusters_k = clusters.copy()
            centroids_k = centroids.copy()
        if nb_rows is None:
            nb_rows = int(np.ceil(k/nb_cols))
        if self.time_points is not None:
            time_points = self.time_points
        else:
            time_points = range(self.nb_time_pts)
        if warps is not None:
            if isinstance(warps, dict):
                warps_k = warps[str(k)]
            else:
                warps_k = warps.copy()
            # Each warp type is displayed with a different color:
            cmap = cm.get_cmap('cubehelix')
            warp_colors = cmap(np.linspace(
                0.25, 0.75, len(np.unique(warps_k))))
        if figsize is None:
            figsize = (15, 6 * nb_rows)
        fig = plt.figure(figsize=figsize)
        axs = fig.subplots(nb_rows, nb_cols, sharey=True)
        # Plot the corresponding cluster in each subplot:
        # (this part is common for the cases with and without warping, in the
        # former case plots unaligned fold changes)
        for j in range(k):
            cluster_j = np.argwhere(clusters_k == j)
            if len(cluster_j) > 1:
                cluster_j = np.squeeze(cluster_j)
            ax_j = axs[j//nb_cols] if (nb_cols > 1) and (nb_rows != 1) else axs
            col_to_plot = j % nb_cols if nb_cols != 1 else j
            for f in range(len(cluster_j)):
                if warps is not None:
                    warp_type = warps_k[cluster_j[f]]
                    c = warp_colors[np.max(warps_k) + warp_type]
                    ax_j[col_to_plot].plot(time_points,
                                           self.means[:, cluster_j[f]],
                                           '-.', color=c)
                else:
                    ax_j[col_to_plot].plot(time_points,
                                           self.means[:, cluster_j[f]],
                                           '-.', color='grey')
            if centroid_type == 'medoid':  # the centroid is colored in red and thick
                ax_j[col_to_plot].plot(time_points, self.means[:, centroids_k[j]],
                                       '-', color='red', linewidth=2,
                                       label='Centroid')
                if self.var_names is not None:
                    cluster_title = f'Centroid: {self.var_names[centroids_k[j]]}, {len(cluster_j)} members'
                    ax_j[col_to_plot].set_title(cluster_title, color='red')
                else:
                    cluster_title = f'{len(cluster_j)} members'
                    ax_j[col_to_plot].set_title(cluster_title, color='red')
            if centroid_type == 'barycenter':
                ax_j[col_to_plot].plot(time_points, centroids_k[:, j], '-',
                                       color='red', linewidth=2,
                                       label='Barycenter')
                cluster_title = f'{len(cluster_j)} members'
                ax_j[col_to_plot].set_title(cluster_title, color='red')
            if warps is not None:  # adding a legend specifying warp types
                lines = [Line2D([0], [0], color=c, linewidth=1,
                                linestyle='-') for c in warp_colors]
                labels = ['' for i in range(len(np.unique(warps_k)))]
                warps_cluster_j = warps_k[cluster_j]
                perc_id = round((np.count_nonzero(warps_cluster_j == 0)-1)*100
                                / len(warps_cluster_j))
                labels[np.max(warps_k)] = f'No warping ({perc_id}%)'
                for step in range(1, np.max(warps_k)+1):
                    perc_minus_step = round(np.count_nonzero(warps_cluster_j == -step)
                                            * 100/len(warps_cluster_j))
                    labels[np.max(
                        warps_k)-step] = f'Warped backwards by {step} ({perc_minus_step}%)'
                    perc_plus_step = round(np.count_nonzero(warps_cluster_j == step)
                                           * 100/len(warps_cluster_j))
                    labels[np.max(
                        warps_k)+step] = f'Warped forward by {step} ({perc_plus_step}%)'
                lines.append(Line2D([0], [0], color='red',
                             linewidth=2, linestyle='-'))
                labels.append('Centroids')
                ax_j[col_to_plot].legend(lines, labels)
                ax_j[col_to_plot].axhline(y=0, xmin=0, xmax=time_points[-1]+1,
                                          linestyle='--', color='k')
        if warps is not None:
            plt.suptitle('Unwarped fold changes', fontsize=20)
        plt.tight_layout()
        plt.show()

        # Second part is for time warping only: shows post-warping clusters
        # (aligned)
        if warps is not None:
            fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize,
                                    sharey=True)
            # Plot the corresponding cluster in each subplot:
            for j in range(k):
                cluster_j = np.argwhere(clusters_k == j)
                if len(cluster_j) > 1:
                    cluster_j = np.squeeze(cluster_j)
                ax_j = axs[j//nb_cols] if (nb_cols >
                                           1) and (nb_rows != 1) else axs
                col_to_plot = j % nb_cols if nb_cols != 1 else j
                for f in range(len(cluster_j)):
                    warp_type = warps_k[cluster_j[f]]
                    warped_range = np.asarray(range(-min(0, warp_type),
                                                    self.nb_time_pts-max(0, warp_type)))
                    plot_range = np.asarray(range(max(0, warp_type),
                                                  self.nb_time_pts+min(0, warp_type)))
                    c = warp_colors[np.max(warps_k) + warp_type]
                    ax_j[col_to_plot].plot(time_points[plot_range],
                                           self.means[warped_range,
                                                      cluster_j[f]],
                                           '-.', color=c)
                ax_j[col_to_plot].plot(time_points, self.means[:, centroids_k[j]],
                                       '-', color='red', linewidth=2, label='Centroid')
                if self.var_names is not None:
                    cluster_title = f'Centroid: {self.var_names[centroids_k[j]]}, {len(cluster_j)} members'
                    ax_j[col_to_plot].set_title(cluster_title, color='red')
                # Adding a legend specifying warp types
                lines = [Line2D([0], [0], color=c, linewidth=1,
                                linestyle='-') for c in warp_colors]
                labels = ['' for i in range(len(np.unique(warps_k)))]
                warps_cluster_j = warps_k[cluster_j]
                perc_id = round((np.count_nonzero(warps_cluster_j == 0)-1)
                                * 100/len(warps_cluster_j))
                labels[np.max(warps_k)] = f'No warping ({perc_id}%)'
                for step in range(1, np.max(warps_k)+1):
                    perc_minus_step = round(np.count_nonzero(warps_cluster_j == -step)
                                            * 100/len(warps_cluster_j))
                    labels[np.max(
                        warps_k)-step] = f'Warped backwards by {step} ({perc_minus_step}%)'
                    perc_plus_step = round(np.count_nonzero(warps_cluster_j == step)
                                           * 100/len(warps_cluster_j))
                    labels[np.max(
                        warps_k)+step] = f'Warped forward by {step} ({perc_plus_step}%)'
                lines.append(Line2D([0], [0], color='red',
                             linewidth=2, linestyle='-'))
                labels.append('Centroids')
                ax_j[col_to_plot].legend(lines, labels)
                ax_j[col_to_plot].axhline(y=0, xmin=0, xmax=time_points[-1]+1,
                                          linestyle='--', color='k')
            plt.suptitle('Warped fold changes', fontsize=20)
            plt.tight_layout()
            plt.show()


class NetworkInference(Clustering):
    """
    A class containing tools for inference of a network of fold changes from
    a dataset, inherits from Clustering and FoldChanges classes.

    Attributes
    ----------
    sparsity : float
        Sparsity of the network determining the cutoff when defining the
        binary adjacency matrix adj_mat based on the weighted one.

    directed : bool
        If True, the network is directed, and undirected if False.

    adj_mat : ndarray
        2D array of shape (nb_var, nb_var) indicating whether the fold changes
        are connected (i.e. similar enough) or not. If the network is
        undirected, then has 0 for connected fold changes and 1 for not
        connected (symmetric). A pair of fold changes is considered to be
        connected if their distance-based similarity is bigger then the cutoff
        value, which is equal to the empirical quantile of the similarity
        matrix corresponding to the chosen sparsity. If the network is directed,
        the matrix stops being symmetric, and the edges that exist according
        to the undirected case procedure become either 1 or 0 based on the
        corresponding warp: 1 for the edges with the corresponding warps being
        positive (predictive) or 0 (simultaneous), and 0 for
        those with negative warps (target).

    Methods
    -------
    infer_sbm(nb_blocks, clusters, n_init=10, n_iter_early_stop=50,
              random=False, verbosity=0, pi_weight=0.8, random_gen=None)
        Performs stochastic block model inference for the fold changes' network
        based on clustering (i.e. on the constrained parameter space).
    compute_network(clusters, centroids, draw_path=False, path=None,
                    figtitle='Fold changes network', figsize=(25,25),
                    obj_scale=1, graph_type='full', adj_mat_2=None,
                    shade_intersect=False)
        Creates a NetworkX object representing the fold changes' network and
        displays it in a block form arising from clusters. The network is
        represented with a graph where nodes are the considered entities and
        the edges are connections between them (i.e. ones in the adjacency
        matrix). Members of every block are grouped around their centroid
        (its node is bigger then other nodes), and have a color different
        from other blocks.
    plot_most_connected_members(clusters, centroids=None, warps=None,
                                nb_components=5)
        Identifies the most connected components within each cluster, and
        displays a plot of the corresponding fold changes' means. If warps are
        given, then also displays the information on the warping groups of
        the components.
    compute_entity_path(path_e1_to_e2=None, entity_1=None, entity_2=None,
                        plot=True)
        If entity_1 and entity_2 are given (and path_e1_to_e2 is not),
        computes a shortest path from entity_1 to entity_2, and plots a figure
        with the means of the fold changes in the path. If path_e1_to_e2 is
        given, then produces a plot of the means of the fold changes'
        in path_e1_to_e2.
    draw_mesoscopic(clusters, centroids, obj_scale=1, node_label_size=30)
        Displays a mesoscopic representation of the fold changes network, i.e.
        a graph with k=len(centroids) nodes representing clusters, each labeled
        by the name of the corresponding centroid, with sizes proportional to
        respective cluster sizes. The edges represent connections between
        clusters, their thickness is proportional to the respective number of
        connections. If the network is directed, then arrow head sizes are
        proportional to the percentage of connections of the corresponding
        predictive type among all connections between the considered clusters.
        In the latter case edges are annotated with the distribution among the
        connection types (i.e. warps) in the following format: for an edge
        between A and B, the annotation is of the form "% of predictive
        connections from B to A - total number of connections between
        A and B - % of predictive connections from A to B". In the case of
        undirected graph, the edges are annotated with the corresponding
        numbers of connections only.
    graph_analysis(clusters, nb_top=10)
        Performs a series of graph analyses of the fold changes network, in
        particular: identifies among the entities nb_top top hits, authorities,
        nodes with respect to pagerank, degree and betweenness centrality. It
        also plots a figure displaying degree distribution of the nodes.
    pathway_search(clusters)
        Identifies all shortest paths between entities in the network of length
        3 and bigger, and presents them along with their scores with respect to
        criteria potentially relevant for hypothesis generation.

    """
    __doc__ += 'Inherited from Clustering: \n ' + Clustering.__doc__

    def __init__(self, data=None, means=None, cov=None, var_names=None,
                 time_points=None, dist='d2hat', time_warp=False,
                 max_warp_step=1, sign_pen=False, pen_param=1, random_gen=None,
                 sparsity=0.75, directed=False, adj_mat=None):
        """
        Parameters
        ----------
        data : ndarray or None
            If not None, 4D array with the dimensions corresponding to:
            1) nb of time points, 2) two experimental conditions
            (dim 0: control, dim 1: case)), 3) replicates, 4) nb of entities.
            If None (by default), then the fold changes are constructed based
            on 'means' and 'cov'. Either 'data' or 'means' and 'cov' have to be
            non-None, with 'data' having priority for the fold changes
            construction.
        means : ndarray or None
            If not None, 2D array of shape (nb_time_pts, nb_var)
            containing data with `float` type, representing fold changes' means
            for each entity and each time point. If 'data' is None, used to
            construct fold changes. Either 'data' or 'means' and 'cov' have to
            be non-None.
        cov : ndarray or None
            If not None, 3D array of shape (nb_time_pts, nb_var, nb_var)
            containing data with `float` type, representing fold changes'
            nb_var x nb_var shaped covariance matrices for each time point.
            Time-wise cross-covariances are assumes to be 0 due to experimental
            design. In case of Hellinger distance, can also be 4-dimensional
            (natural form): (nb_time_pts, nb_time_pts, nb_var, nb_var).
            If 'data' is None, used to construct fold changes. Either 'data' or
            'means' and 'cov' have to be non-None.
        var_names : array-like or None
            1D array-like containing data with `string` type, representing
            names of the measured entities (ex. genes). The default is None.
        time_points : array-like or None
            1D array-like containing data with `float` type, representing time
            points at which fold changes were measured. The default is None.
        dist : str
            Distance chosen for clustering, 'd2hat' by default (L2 distance
            between random estimators), can also be 'wasserstein'
            (Wasserstein distance) and 'hellinger' (Hellinger distance).
        time_warp : bool
            If True, then the clustering procedure is coupled with the
            alignment. The default is False.
        max_warp_step : int
            If max_warp_step=i>0, then the set of all considered warps is the
            set of all integers between -i and i.
        sign_pen : bool
            If True, then the distance is penalized with sign penalty.
            The default is False.
        pen_param : float
            Parameter determining the weight of sign penalty. The default is 1.
        random_gen : RandomState instance or None
            Random number generator, used to reproduce results. If None
            (default), the generator is the RandomState instance used by
            `np.random`. If RandomState instance, random_gen is the actual
            random number generator.
        sparsity : float, optional
            Sparsity of the network determining the cutoff when defining the
            binary adjacency matrix based on the weighted one.
            The default is 0.75.
        directed : bool, optional
            If True, the network is directed, and undirected if False (default).
        adj_mat : ndarray or None, optional
            If not None (default), 2D array of shape (nb_var, nb_var)
            indicating whether the fold changes are connected (i.e. similar
            enough) or not. If the network is undirected, then has 0 for
            connected fold changes and 1 for not connected (symmetric).
            A pair of fold changes is considered to be connected if their
            distance-based similarity is bigger then the cutoff value, which
            is equal to the empirical quantile of the similarity matrix
            corresponding to the chosen sparsity. If the network is directed,
            the matrix stops being symmetric, and the edges that exist
            according to the undirected case procedure become either 1 or 0
            based on the corresponding warp: 1 for the edges with the
            corresponding warps being positive (predictive) or 0
            (simultaneous), and 0 for those with negative warps (target).
            If 'adj_mat' is specified, the adjacency matrix is defined based 
            its value, otherwise calculated based on the distance matrix and 
            the optimal distance matrix. NB: in the former case 'optimal_warp_mat'
            is recalculated to correspond to 'adj_mat', however 'dist_mat'
            remains the same.
        Returns
        -------
        None.

        """
        super().__init__(data=data, means=means, cov=cov, var_names=var_names,
                         time_points=time_points, dist=dist,
                         time_warp=time_warp, max_warp_step=max_warp_step,
                         sign_pen=sign_pen, pen_param=pen_param,
                         random_gen=random_gen)
        self.sparsity = sparsity
        self.directed = directed
        if adj_mat is not None:
            self.adj_mat = adj_mat
            if not self.directed:
                is_sym = np.array_equal(self.adj_mat, self.adj_mat.T)
                assert is_sym, "Adjacency matrix for an undirected graph has to be symmetric."
            else:
                self.optimal_warp_mat[(self.adj_mat == 1) 
                                      & (self.adj_mat.T == 1)] = 0
                self.optimal_warp_mat[(self.adj_mat == 1) 
                                      & (self.adj_mat.T == 0)] = 1
                self.optimal_warp_mat[(self.adj_mat == 0) 
                                      & (self.adj_mat.T == 1)] = -1
        # Calculating adjacency matrix based on the distance matrix:
        elif self.dist_mat is not None:
            max_dist = np.max(self.dist_mat)
            # Normalizing distance to get similarity matrix:
            simil_mat = (-self.dist_mat+max_dist)/max_dist
            simil_flat = (-self.distances+max_dist)/max_dist
            # Matrix binarization through sparsity-based thresholding:
            sparsity_quantile = np.quantile(simil_flat, sparsity)
            sparse_simil_array = np.where(simil_mat >= sparsity_quantile, 1, 0)
            sparse_simil_array[np.diag_indices(self.nb_var)] = 0
            # If directed, keep the edges with positive warps:
            if self.directed and self.time_warp:
                self.adj_mat = np.where(self.optimal_warp_mat < 0, 0,
                                        sparse_simil_array)
            else:
                self.adj_mat = sparse_simil_array.copy()

    def infer_sbm(self, nb_blocks, clusters, n_init=10, n_iter_early_stop=50,
                  random=False, verbosity=0, pi_weight=0.8, random_gen=None):
        """
        Performs stochastic block model inference for the fold changes' network
        based on clustering (i.e. on the constrained parameter space). This
        code is based on method 'fit' from class 'SBM' of the package SparseBM
        (https://github.com/gfrisch/sparsebm).

        Parameters
        ----------
        nb_blocks : int
            Number of blocks (communities/clusters) in the stochastic
            block model.
        clusters : ndarray or dictionary
            If ndarray, 1D array of length nb_var containing integers in range
            (0, k) indicating clusters to which the fold changes are assigned.
            If a dictionary, the keys are numbers of clusters considered, and
            for each such number the value is the latter array.
        n_init : int, optional
            Number of initializations. The default is 10.
        n_iter_early_stop : TYPE, optional
            Number of VEM iterations. The default is 50.
        random : bool, optional
            If True, stochastic block model is initialized on the parameter
            space defined by the original model. If False (default),
            stochastic block model is initialized on the constrained parameter
            space corresponding to base clustering.
        verbosity : int, optional
            Degree of verbosity. Scale from 0 (no message displayed) to 3.
            The default is 0.
        pi_weight : float, optional
            Weight parameter controlling the initialization of pi.
            pi(q,q)~Unif([pi_weight, 1)) and pi(q,q')~Unif([0,1-pi_weight))
            for q!=q'. The default is 0.8.
        random_gen : RandomState instance or None, optional
            Random number generator, used to reproduce results. If None
            (default), the generator is the RandomState instance used by
            `np.random`. If RandomState instance, random_gen is the actual
            random number generator.

        Returns
        -------
        successful_sbm : SBM instance or None
            Successfully trained stochastic block model, or None in case of
            failure.
,        sbm_centroids : ndarray or None
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that have been chosen as centroids (calculated
             after stochastic block model is inferred) if SBM is successfully
             inferred, otherwise None.
        comp_cost : float or None
            Value of the total comparable cost if SBM is successfully inferred,
            otherwise None.

        """
        eps = 1e-2 / self.nb_var
        sbm = SBM(nb_blocks, n_iter_early_stop=n_iter_early_stop,
                  verbosity=verbosity)
        sbm._nb_rows = nb_blocks
        sbm.symmetric = True
        sbm._check_params()
        sparse_simil_mat = sparse.csr_matrix(self.adj_mat)

        comp_cost = np.inf
        counter = 0
        was_ever_successful = False

        if random_gen is None:
            random_gen = np.random

        if isinstance(clusters, dict):
            clusters_k = clusters[str(nb_blocks)]
        else:
            clusters_k = clusters.copy()
        for init in range(n_init):
            # Case without additional constraints:
            if random:
                eps = 1e-2 / self.nb_var

                alpha_init = ((np.ones(nb_blocks) / nb_blocks)
                              .reshape((nb_blocks, 1)))
                alpha_init = alpha_init.flatten()

                tau_init = random_gen.uniform(size=(self.nb_var,
                                                    nb_blocks)) ** 2
                tau_init /= tau_init.sum(axis=1).reshape(self.nb_var, 1)
                tau_init[tau_init < eps] = eps
                tau_init /= (tau_init.sum(axis=1)
                             .reshape(self.nb_var, 1))  # Re-Normalize.
                pi_init = random_gen.uniform(2 * sparse_simil_mat.nnz
                                             / (self.nb_var * self.nb_var) / 10,
                                             2 * sparse_simil_mat.nnz
                                             / (self.nb_var * self.nb_var),
                                             (nb_blocks, nb_blocks))
                indices_ones = list(self.adj_mat.nonzero())

                # VEM:
                (success, ll,
                 pi, alpha, tau) = sbm._fit_single(self.adj_mat, indices_ones,
                                                   self.nb_var, run_number=1,
                                                   init_params=(pi_init,
                                                                alpha_init,
                                                                tau_init),
                                                   early_stop=sbm.n_iter_early_stop)
                if success:
                    sbm.trained_successfully_ = True
                    was_ever_successful = True
                    sbm_clusters = tau.argmax(1)
                    comp_cost_sbm = self.calculate_comparable_cost(nb_blocks,
                                                                   sbm_clusters)
                    if ll > sbm.loglikelihood_:
                        counter += 1
                        sbm.loglikelihood_ = ll
                        comp_cost = comp_cost_sbm
                        sbm.pi_ = pi
                        sbm.alpha_ = alpha
                        sbm.tau_ = tau
                        sbm_centroids = self.hierarchical_centroids(nb_blocks,
                                                                    sbm_clusters)
                        if np.isnan(sbm_centroids).all():
                            existing_clusters = np.unique(sbm_clusters)
                            new_nb_blocks = len(existing_clusters)
                            sbm_centroids = np.repeat(np.nan, nb_blocks)
                            sbm_centroids[existing_clusters] = self.hierarchical_centroids(new_nb_blocks,
                                                                                           sbm_clusters)
                        successful_sbm = sbm.copy()
            # Case with clustering related constraints:
            else:
                alpha_init = np.ones(nb_blocks)/nb_blocks
                # Initialization of the variational latent variable
                # distribution parameter tau according to k-medoids clustering:
                tau_init = np.ones((self.nb_var, nb_blocks)) * eps
                for i, cl in enumerate(clusters_k):
                    tau_init[i, cl] = 1 - eps * (nb_blocks-1)
                # Initialization of the parameter determining edges to
                # correspond to hard clustering:
                pi_init = random_gen.uniform(0, 1-pi_weight,
                                             (nb_blocks, nb_blocks))
                pi_init[np.diag_indices(nb_blocks)] = random_gen.uniform(pi_weight,
                                                                         1,
                                                                         (1, nb_blocks))
                pi_init[np.triu_indices(nb_blocks, 1)[
                    ::-1]] = pi_init[np.triu_indices(nb_blocks, 1)]

                indices_ones = list(self.adj_mat.nonzero())

                # VEM:
                (success, ll,
                 pi, alpha, tau) = sbm._fit_single(self.adj_mat, indices_ones,
                                                   self.nb_var, run_number=1,
                                                   init_params=(pi_init,
                                                                alpha_init,
                                                                tau_init),
                                                   early_stop=sbm.n_iter_early_stop)
                if success:
                    sbm.trained_successfully_ = True
                    was_ever_successful = True
                    sbm_clusters = tau.argmax(1)
                    comp_cost_sbm = self.calculate_comparable_cost(nb_blocks,
                                                                   sbm_clusters)
                    if ll > sbm.loglikelihood_ and comp_cost_sbm < comp_cost:
                        counter += 1
                        sbm.loglikelihood_ = ll
                        comp_cost = comp_cost_sbm
                        sbm.pi_ = pi
                        sbm.alpha_ = alpha
                        sbm.tau_ = tau
                        sbm_centroids = self.hierarchical_centroids(nb_blocks,
                                                                    sbm_clusters)
                        successful_sbm = sbm.copy()
        if was_ever_successful:
            return successful_sbm, sbm_centroids, comp_cost
        return (np.nan, np.nan, np.nan)

    def compute_network(self, clusters, centroids, draw_path=False, path=None,
                        figsize=(25, 25), obj_scale=1, graph_type='full',
                        adj_mat_2=None, clusters_2=None, centroids_2=None,
                        shade_intersect=False, degree_view=False):
        """
        Creates a NetworkX object representing the fold changes' network and
        displays it in a block form arising from clusters. The network is
        represented with a graph where nodes are the considered entities and
        the edges are connections between them (i.e. ones in the adjacency
        matrix). Members of every block are grouped around their centroid
        (its node is bigger then other nodes), and have a color different
        from other blocks.

        Parameters
        ----------
        clusters : ndarray
            1D array of length nb_var containing integers indicating clusters
            to which the fold changes are assigned.
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as centroids.
        draw_path : bool, optional
            False by default, if True and the path is given then the path is
            displayed on the graph with red nodes and thick red edges with the
            remaining edges thin and colored in light grey (the remaining
            nodes are displayed normally).
        path : array-like or None, optional
            If not None (default), 1D container with strings (elements should
            belong to var_names) containing names of the entities as nodes
            in the path of interest (in the correct order).
        figsize : (float, float), optional
            Width and height of the figure(s). The default is (25,25).
        obj_scale : float, optional
            Parameter used to control the scale of objects in the graph, which
            zooms in if bigger than 1 and zooms out if smaller than 1.
            The default is 1.
        graph_type : str, optional
            The following options are possible:
                - 'full' (default) : the whole graph is displayed, with edges
                colored in black if undirected, and grey for simultaneous
                and green for predictive connections if directed.
                - 'intersection' : if adj_mat_2 is given, displays only the
                intersection between the main network and the network defined
                by adj_mat_2.
                - 'difference' : if adj_mat_2 is given, displays the main
                network without its intersection with the network defined
                by adj_mat_2.
        adj_mat_2 : ndarray or None, optional
            If not None (default), 2D array of shape (nb_var, nb_var)
            indicating whether the fold changes are connected or not (same
            to adj_mat). Represents the adjacency matrix of some other set of
            fold changes of interest. Should be based on the measurements for
            the same entities as the base network for a proper comparison.
            Used if graph_type is 'intersection' or 'difference'.
        clusters_2 : ndarray, optional
            If not None (default), 1D array of length nb_var containing 
            integers indicating clusters to which the fold changes are assigned.
            This alternative clustering specification serves to color the nodes
            with respect to the corresponding clustering (typically to compare
            clusters to clusters_2).
        centroids_2 : ndarray, optional
            If not None (default), 1D array of length k containing indices in 
            range (0, nb_var) of the fold changes that act as centroids. This
            second sets of centroids associated with an alternative clustering
            clusters_2 is used only for centroid node sizes (typically to 
            compare centroids to centroids_2).
        shade_intersect : bool, optional
            If True, adj_mat_2 is given, and graph_type is 'full' (makes no
            difference if 'intersection' or 'difference'), displays the entire
            graph but shades the intersection by coloring in lightgrey the
            nodes and the edges that belong entirely to the intersection with
            the network defined by adj_mat_2. The default is False.
        degree_view : bool, optional
            If True, the sizes of nodes reflect their degrees (the relationship
            is increasing and non-linear). Otherwise (default), all nodes have 
            the samesizes, except for the centroids that are bigger then the 
            others.

        Returns
        -------
        None.

        """
        if (clusters_2 is not None) and (centroids_2 is not None):
            assert_cond = ((len(centroids_2) == len(centroids)) 
                           and (len(np.unique(clusters_2)) == len(np.unique(clusters_2))))
            assert_str = "Number of clusters in the second set should match that in the first."
            assert assert_cond, assert_str
        sparse_simil_mat = sparse.csr_matrix(self.adj_mat)
        # Extracting edges:
        indices_ones_graph = list(sparse_simil_mat.nonzero())
        # Node sizes:
        if degree_view:
            node_degrees = self.adj_mat + self.adj_mat.T
            node_degrees[node_degrees == 2] = 1
            node_degrees = node_degrees.sum(axis=0)
            node_size = np.exp(1 + 5 * node_degrees /
                               node_degrees.max()) * 20 * obj_scale
        else:
            node_size = np.ones(self.nb_var) * 1000 * obj_scale
            centroids_size = centroids_2 if centroids_2 is not None else centroids
            node_size[centroids_size] = 3500 * obj_scale
        # Create a data frame for node characteristics:
        graph_carac = pd.DataFrame(
            {'gene': self.var_names, 'node size': node_size})
        if shade_intersect and (adj_mat_2 is not None):
            graph_carac['intersect'] = [
                (self.adj_mat[i, :] == adj_mat_2[i, :]).all() for i in range(self.nb_var)]
        else:
            graph_carac['intersect'] = False

        cmap = cm.get_cmap('gist_rainbow')
        cl_colors = cmap(np.linspace(0.15, 1, clusters.max() + 1))
        clusters_color = clusters_2 if clusters_2 is not None else clusters
        # Shading nodes in the intersection (if relevant):
        node_color = ['lightgrey' if (graph_carac['intersect']
                                      .iloc[i]) 
                      else cl_colors[clusters_color[i]] for i in range(self.nb_var)]
        graph_carac['color'] = node_color
        ecolor_main = 'black'
        eweight_main = 0.2

        if self.directed and self.optimal_warp_mat is not None:
            ecolor_dir = 'green'
            eweight_dir = 0.3
            optimal_warp_flat = self.optimal_warp_mat[indices_ones_graph[0],
                                                      indices_ones_graph[1]]
            fc_graph = nx.DiGraph()
            all_edges = np.array(indices_ones_graph).T
            dir_edges = all_edges[(optimal_warp_flat == 1).nonzero()[0]]
            undir_edges = all_edges[(optimal_warp_flat == 0).nonzero()[0]]
            # Constructing the path (if relevant):
            if draw_path and path is not None:
                path_edges = []
                for i, e_1 in enumerate(path[:-1]):
                    e_2 = path[i+1]
                    ind_e_1 = int((self.var_names == e_1).nonzero()[0])
                    ind_e_2 = int((self.var_names == e_2).nonzero()[0])
                    path_edge = [ind_e_1, ind_e_2]
                    path_edges.append(path_edge)
                    dir_edges = np.delete(dir_edges,
                                          ((dir_edges == path_edge)
                                           .all(axis=1)).nonzero(),
                                          axis=0)
                    undir_edges = np.delete(undir_edges,
                                            ((undir_edges == path_edge)
                                             .all(axis=1)).nonzero(),
                                            axis=0)
                    undir_edges = np.delete(undir_edges,
                                            ((undir_edges == path_edge[::-1])
                                             .all(axis=1)).nonzero(),
                                            axis=0)
                    graph_carac['color'].where(graph_carac['gene'] != e_1,
                                               'red', inplace=True)
                graph_carac['color'].where(graph_carac['gene'] != path[-1],
                                           'red', inplace=True)
                eweight_path = 10
                ecolor_path = 'red'
            # Defining characteristics for the cases with intersection, without
            # intersection and with shaded difference:
            if (adj_mat_2 is not None) and (shade_intersect or graph_type != 'full'):
                ecolor_inter_undir = (ecolor_main if graph_type == 'intersection'
                                      else 'lightgrey')
                ecolor_inter_dir = (ecolor_dir if graph_type == 'intersection'
                                    else 'lightgrey')
                eweight_inter_dir = (eweight_dir if graph_type == 'intersection'
                                     else eweight_main)
                sparse_simil_mat_2 = sparse.csr_matrix(adj_mat_2)
                indices_ones_graph_2 = list(sparse_simil_mat_2.nonzero())
                all_edges_2 = np.array(indices_ones_graph_2).T
                dir_edges_zl = list(zip(dir_edges[:, 0], dir_edges[:, 1]))
                undir_edges_zl = list(
                    zip(undir_edges[:, 0], undir_edges[:, 1]))
                all_edges_2_zl = list(
                    zip(all_edges_2[:, 0], all_edges_2[:, 1]))
                dir_to_del = [i for i in range(dir_edges.shape[0]) if (
                    dir_edges_zl[i] in all_edges_2_zl)]
                dir_edges_no_int = np.delete(dir_edges, dir_to_del, axis=0)
                undir_to_del = [i for i in range(undir_edges.shape[0]) if (
                    undir_edges_zl[i] in all_edges_2_zl)]
                undir_edges_no_int = np.delete(
                    undir_edges, undir_to_del, axis=0)
                # Edges for the cases without intersection and with shaded difference:
                if graph_type != 'intersection':
                    fc_graph.add_edges_from(undir_edges_no_int, color=ecolor_main,
                                            weight=eweight_main)
                    fc_graph.add_edges_from(undir_edges_no_int[:, (1, 0)],
                                            color=ecolor_main, weight=eweight_main)
                    fc_graph.add_edges_from(dir_edges_no_int, color=ecolor_dir,
                                            weight=eweight_dir)
                # Edges for the cases with intersection and with shaded difference:
                if graph_type != 'difference':
                    fc_graph.add_edges_from(undir_edges[undir_to_del],
                                            color=ecolor_inter_undir,
                                            weight=eweight_main)
                    fc_graph.add_edges_from(dir_edges[dir_to_del],
                                            color=ecolor_inter_dir,
                                            weight=eweight_inter_dir)
                # Adding the path to the graph (if relevant):
                if draw_path and path is not None:
                    ecolor_inter_path = (ecolor_path if graph_type == 'intersection'
                                         else 'lightgrey')
                    eweight_inter_path = (eweight_path if graph_type == 'intersection'
                                          else eweight_main)
                    path_edges_a = np.array(path_edges)
                    path_edges_zl = list(
                        zip(path_edges_a[:, 0], path_edges_a[:, 1]))
                    path_to_del = [i for i in range(path_edges_a.shape[0]) if (
                        path_edges_zl[i] in all_edges_2_zl)]
                    path_edges_no_int = np.delete(
                        path_edges_a, path_to_del, axis=0)
                    if graph_type != 'intersection':
                        fc_graph.add_edges_from(path_edges_no_int,
                                                color=ecolor_path,
                                                weight=eweight_path)
                    if graph_type != 'difference':
                        fc_graph.add_edges_from(path_edges_a[path_to_del],
                                                color=ecolor_inter_path,
                                                weight=eweight_inter_path)
            else:
                # Adding the path to the graph (if relevant):
                if draw_path and path is not None:
                    ecolor_dir = 'grey'
                    ecolor_main = 'grey'
                    eweight_dir = eweight_main
                    fc_graph.add_edges_from(path_edges, color='red',
                                            weight=eweight_path)
                fc_graph.add_edges_from(undir_edges, color=ecolor_main,
                                        weight=eweight_main)
                fc_graph.add_edges_from(undir_edges[:, (1, 0)], color=ecolor_main,
                                        weight=eweight_main)
                fc_graph.add_edges_from(dir_edges, color=ecolor_dir,
                                        weight=eweight_dir)
        else:  # Undirected graph
            fc_graph = nx.Graph()
            all_edges = np.array(indices_ones_graph).T
            fc_graph.add_edges_from(all_edges, color=ecolor_main,
                                    weight=eweight_main)

        nodes_dict = {}
        for i in range(self.nb_var):
            nodes_dict[i] = self.var_names[i]
        fc_graph = nx.relabel_nodes(fc_graph, nodes_dict)

        graph_carac = graph_carac.set_index('gene')
        graph_carac = graph_carac.reindex(fc_graph.nodes())
        # Assigning positions to centroids:
        nb_blocks = len(centroids)
        theta = np.linspace(0, 1, len(centroids) + 1)[:-1] * 2 * np.pi
        theta = theta.astype(np.float32)
        centroids_pos = np.column_stack([np.cos(theta), np.sin(theta),
                                         np.zeros((len(centroids), 0))])
        centroids_pos = nx.rescale_layout(centroids_pos, scale=1.3)
        centroids_pos = dict(zip(self.var_names[centroids], centroids_pos))
        centroids_pos_df = pd.DataFrame(centroids_pos).sort_values(0, axis=1)
        centroids_pos_df.columns = self.var_names[centroids]
        centroids_pos = centroids_pos_df.to_dict('list')
        fc_positions = centroids_pos.copy()
        missing = []
        # Creating subgraphs for every cluster:
        for i in range(nb_blocks):
            centroid_i = self.var_names[centroids][i]
            cluster_i = np.argwhere(clusters == i).squeeze()
            cluster_i_rest = np.delete(self.var_names[cluster_i],
                                       np.where(self.var_names[cluster_i] == centroid_i))
            cluster_i_subgraph = fc_graph.subgraph(cluster_i_rest)
            subnet_scale = 1.2 * cluster_i.size/self.nb_var + 3 / nb_blocks
            cluster_i_layout = nx.kamada_kawai_layout(cluster_i_subgraph,
                                                      center=centroids_pos[centroid_i],
                                                      scale=subnet_scale)
            # Missing (isolated) nodes are ignored in the graph and printed
            # for the user:
            for g in cluster_i_rest:
                try:
                    fc_positions[g] = cluster_i_layout[g]
                except KeyError:
                    missing.append(g)
                    print(f'{g} is missing')
        edge_colors = nx.get_edge_attributes(fc_graph, 'color').values()
        edge_widths = nx.get_edge_attributes(fc_graph, 'weight').values()

        plt.figure(figsize=figsize)
        nx.draw_networkx(fc_graph, fc_positions, with_labels=True, 
                         width=list(edge_widths), node_color=graph_carac['color'], 
                         font_size=8*obj_scale, node_size=graph_carac['node size'], 
                         edge_color=edge_colors, alpha=0.5, font_family='serif')
        plt.margins(0.0)
        plt.show()
        return

    def plot_most_connected_members(self, clusters, centroids=None, warps=None,
                                    nb_components=5, figsize=None):
        """
        Identifies the most connected components within each cluster, and
        displays a plot of the corresponding fold changes' means. If warps are
        given, then also displays the information on the warping groups of
        the components.

        Parameters
        ----------
        clusters : ndarray
            1D array of length nb_var containing integers indicating clusters
            to which the fold changes are assigned.
        centroids : ndarray or None, optional
            If not None (default), 1D array of length k containing indices in
            range (0, nb_var) of the fold changes that act as centroids.
        warps : ndarray or None, optional
            If not None (default), 1D array of length nb_var containing
            integers in range (-max_warp_step, max_warp_step + 1)
            indicating fold changes' warps with respect to their
            corresponding centroids.
        nb_components : int, optional
            Number of the most connected components to select in each cluster.
            The default is 5.
        figsize : (float, float), optional
            Width and height of the figure(s). The default is None.

        Returns
        -------
        most_connected_members_within : ndarray
            2D array of shape (nb_blocks, nb_components) containing indices
            in range (0, nb_var) of nb_components most connected components
            for each cluster (block).

        """
        nb_blocks = clusters.max()+1
        most_connected_members_within = np.zeros((nb_blocks, nb_components),
                                                 dtype=int)
        ht_ratios = np.ones(nb_blocks+1)
        ht_ratios[0] *= 0.02
        warp_type_lines = list(mpl.lines.lineStyles.keys())[:4]
        warp_type_lines.extend(list(mpl.markers.MarkerStyle.markers.keys()))
        # covers up to 45 warp types (should be enough)

        mpl.rcParams['lines.linewidth'] = 3
        sns.set_style("darkgrid")
        if figsize is None:
            figsize = (10, nb_blocks * 6)
        fig, axs = plt.subplots(nb_blocks + 1, 1,
                                figsize=figsize, sharey=False,
                                gridspec_kw={"height_ratios": ht_ratios})
        label_scale = min(figsize) / 10
        for i in range(nb_blocks):
            #ax_i = axs[i//nb_cols+1] if (nb_cols > 1) and (nb_rows != 1) else axs
            #col_to_plot = i % nb_cols if nb_cols != 1 else i
            if centroids is not None:
                centroid_str = self.var_names[centroids[i]]
                print(f'Cluster {i+1}: ' + centroid_str)
            else:
                print('Cluster {i+1}')
            print('Most connected members within cluster:')
            cluster_i = np.argwhere(clusters == i).squeeze()
            simil_array_cluster_i = self.adj_mat[cluster_i, :][:, cluster_i]
            most_connected_members_within[i, :] = cluster_i[np.argsort(np.sum(simil_array_cluster_i,
                                                                              axis=1))[-nb_components:]]
            legend_i = list(
                self.var_names[most_connected_members_within[i, :]])
            print(legend_i)
            for it in range(nb_components):
                if self.time_warp and warps is not None:
                    warp_it = warps[most_connected_members_within[i, it]]
                    legend_i[it] += f" (warped by {warp_it})"
                    linestyle_it = warp_type_lines[warps.max() + warp_it]
                else:
                    linestyle_it = '-'
                axs[i+1].plot(self.time_points,
                              self.means[:,
                                         most_connected_members_within[i, :][it]],
                              linestyle_it)
            if centroids is not None:
                # Centroids used as subtitles corresponding to clusters:
                axs[i+1].set_title('Centroid: ' + centroid_str, color='red')
                axs[i+1].legend(legend_i, fontsize=14 * label_scale, 
                                labelspacing=0.2 * label_scale)
            axs[i+1].axhline(y=0, xmin=0, xmax=self.time_points[-1]+1,
                             linestyle='--', color='k')
        axs[0].axis("off")
        axs[0].set_title(f"{nb_components} most connected \n members within clusters",
                         fontweight='bold', fontsize=20 * label_scale)
        fig.tight_layout()
        plt.show()
        return most_connected_members_within

    def compute_entity_path(self, path_e1_to_e2=None, entity_1=None,
                            entity_2=None, plot=True, figsize=(10, 7)):
        """
        If entity_1 and entity_2 are given (and path_e1_to_e2 is not),
        computes a shortest path from entity_1 to entity_2, and plots a figure
        with the means of the fold changes in the path. If path_e1_to_e2 is
        given, then produces a plot of the means of the fold changes'
        in path_e1_to_e2.

        Parameters
        ----------
        path_e1_to_e2 : array-like or None, optional
            If not None (default), 1D container with strings (elements should
            belong to var_names) containing names of the entities as nodes
            in the path of interest (in the correct order).
            Either 'path_e1_to_e2' or 'entity_1' and 'entity_1' have to be
            non-None, with 'path_e1_to_e2' having priority for the path
            construction.
        entity_1 : str or None, optional
           Starting node for path. The default is None.
           Either 'path_e1_to_e2' or 'entity_1' and 'entity_1' have to be
           non-None, with 'path_e1_to_e2' having priority for the path
           construction.
        entity_2 : str or None, optional
            Ending node for path. The default is None.
            Either 'path_e1_to_e2' or 'entity_1' and 'entity_1' have to be
            non-None, with 'path_e1_to_e2' having priority for the path
            construction.
        plot : bool, optional
            If True (default), displays a figure with the means of the fold
            changes in the path.
        figsize : (float, float), optional
            Width and height of the figure(s). The default is (10,7).

        Returns
        -------
        path_e1_to_e2 : array-like
            1D container with strings containing names of the entities as nodes
            in the path of interest.
        path_e1_to_e2_warps : list
            Contains len(path_e1_to_e2)-1 elements, the warps between the
            consecutive nodes in the path, allows to determine the extend to
            which the path has a predictive character. Returned if the graph
            is directed.

        """
        assert_str = "No path or entities to connect provided."
        assert (path_e1_to_e2 is not None) or (
            entity_1 is not None and entity_2 is not None), assert_str

        # Create a NetworkX object:
        fc_graph = nx.DiGraph(self.adj_mat)
        nx.relabel_nodes(fc_graph, dict(enumerate(self.var_names)), copy=False)

        # Compute shortest path:
        if path_e1_to_e2 is None:
            path_e1_to_e2 = nx.shortest_path(fc_graph, entity_1, entity_2)
        path_e1_to_e2_warps = []
        s_path = "Gene path: "
        s_warps = "Warps: "

        if plot:
            mpl.style.use('seaborn')
            mpl.rcParams['font.family'] = 'serif'
            fig, ax = plt.subplots(figsize=figsize)
            cmap = cm.get_cmap('gist_rainbow')
            curve_colors = cmap(np.linspace(0, 1, len(path_e1_to_e2)))
        for i, e in enumerate(path_e1_to_e2[:-1]):
            ind_e_1 = np.argwhere(self.var_names == e).squeeze()
            ind_e_2 = np.argwhere(
                self.var_names == path_e1_to_e2[i+1]).squeeze()
            s_path += f"{e} --> "
            # Extract warps:
            if self.time_warp:
                warp_e1_to_e2 = self.optimal_warp_mat[ind_e_1, ind_e_2]
                path_e1_to_e2_warps.append(warp_e1_to_e2)
                s_warps += f"  {warp_e1_to_e2} "
            if plot:
                ax.plot(self.time_points, self.means[:, ind_e_1],
                        c=curve_colors[i])
        s_path += (entity_2 if entity_2 is not None else path_e1_to_e2[-1])

        # Plot curves:
        if plot:
            ax.plot(self.time_points, self.means[:, ind_e_2],
                    c=curve_colors[-1])
            ax.legend(path_e1_to_e2, fontsize=12, labelspacing=0.2)
            ax.set_title(s_path, fontweight='bold', fontsize=16)
            fig.tight_layout()
            plt.show()
        print(s_path)
        if self.time_warp:
            print(s_warps)
            return path_e1_to_e2, path_e1_to_e2_warps
        return path_e1_to_e2

    def draw_mesoscopic(self, clusters, centroids, obj_scale=1,
                        node_label_size=30, figsize=(20, 20)):
        """
        Displays a mesoscopic representation of the fold changes network, i.e.
        a graph with k=len(centroids) nodes representing clusters, each labeled
        by the name of the corresponding centroid, with sizes proportional to
        respective cluster sizes. The edges represent connections between
        clusters, their thickness is proportional to the respective number of
        connections. If the network is directed, then arrow head sizes are
        proportional to the percentage of connections of the corresponding
        predictive type among all connections between the considered clusters.
        In the latter case edges are annotated with the distribution among the
        connection types (i.e. warps) in the following format: for an edge
        between A and B, the annotation is of the form "% of predictive
        connections from B to A - total number of connections between
        A and B - % of predictive connections from A to B". In the case of
        undirected graph, the edges are annotated with the corresponding
        numbers of connections only.

        Parameters
        ----------
        clusters : ndarray
            1D array of length nb_var containing integers indicating clusters
            to which the fold changes are assigned.
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as centroids.
        obj_scale : float, optional
            Parameter used to control the scale of objects in the graph,
            which zooms in if bigger than 1 and zooms out if smaller than 1.
            The default is 1.
        node_label_size : int, optional
            Font size for text labels on nodes (names of centroids).
            The default is 30.
        figsize : (float, float), optional
            Width and height of the figure(s). The default is (20,20).

        Returns
        -------
        None.

        """
        nb_blocks = len(centroids)
        node_size = np.array([(np.count_nonzero(clusters == i)
                               / self.nb_var) for i in range(nb_blocks)])
        clust_pairs = np.array(
            list(itertools.combinations(range(nb_blocks), 2)))
        clust_pairs_tuple = []
        # Array containing numbers of connections for each pair of clusters
        # --> edge widths:
        nb_connect_mat = np.zeros((nb_blocks, nb_blocks), dtype=int)
        edge_labels = {}
        scale = obj_scale * 18

        # Directed case (with time warping), i.e. with arrows and annotations:
        if self.directed and self.optimal_warp_mat is not None:
            meso_graph = nx.DiGraph()
            prop_warp_mat = np.zeros((nb_blocks, nb_blocks))
            for (i, j) in clust_pairs:
                clust_pairs_tuple.append((i, j))

                # Block of the adjacency matrix corresponding to connections
                # from fold changes in cluster i to those in j:
                clusters_i_j_adj_mat = self.adj_mat[clusters ==
                                                    i, :][:, clusters == j]
                # Adding connections from i to j:
                nb_connect_mat[i, j] = clusters_i_j_adj_mat.sum()

                # Corresponding block of the OW matrix:
                clusters_i_j_warpmat = self.optimal_warp_mat[clusters == i,
                                                             :][:, clusters == j]
                # Warps of existing connections:
                warps_of_connections_ij = clusters_i_j_warpmat[clusters_i_j_adj_mat
                                                               .astype(bool)]

                # Block of the adjacency matrix corresponding to connections
                # from fold changes in cluster j to those in i:
                clusters_j_i_adj_mat = self.adj_mat[clusters == j, :][:,
                                                                      clusters == i]
                # Corresponding block of the OW matrix:
                clusters_j_i_warpmat = self.optimal_warp_mat[clusters == j,
                                                             :][:, clusters == i]
                warps_of_connections_ji = clusters_j_i_warpmat[clusters_j_i_adj_mat
                                                               .astype(bool)]
                # Adding connections from j to i:
                nb_connect_mat[i, j] += (clusters_j_i_adj_mat.sum()
                                         - np.count_nonzero(warps_of_connections_ij == 0))
                # Calculate proportions of each warp type --> arrow head sizes:
                prop_warp_mat[i, j] = (np.count_nonzero(warps_of_connections_ij > 0) /
                                       (nb_connect_mat[i, j] if nb_connect_mat[i, j] != 0 else 1))

                prop_warp_mat[j, i] = (np.count_nonzero(warps_of_connections_ji > 0) /
                                       (nb_connect_mat[i, j] if nb_connect_mat[i, j] != 0 else 1))
                meso_graph.add_edge(i, j,
                                    width=nb_connect_mat[i, j] * obj_scale**2,
                                    arrowsize=prop_warp_mat[i, j])
                meso_graph.add_edge(j, i,
                                    width=nb_connect_mat[i, j] * obj_scale**2,
                                    arrowsize=prop_warp_mat[j, i])
                # Annotation:
                if nb_connect_mat[i, j] != 0:
                    edge_labels[(i, j)
                                ] = f"{round(prop_warp_mat[j, i] * 100)}%"
                    edge_labels[(i, j)] += f"-{nb_connect_mat[i, j]}"
                    edge_labels[(i, j)
                                ] += f"-{round(prop_warp_mat[i, j] * 100)}%"

            node_labels = {n: self.var_names[centroids[n]] for n in meso_graph}
            sns.set_style("white")

            # Plot the graph:
            plt.figure(figsize=figsize)
            theta = np.linspace(0, 1, nb_blocks + 1)[:-1] * 2 * np.pi
            theta = theta.astype(np.float32)
            pos = np.column_stack([np.cos(theta), np.sin(theta),
                                   np.zeros((nb_blocks, 0))])
            pos = nx.rescale_layout(pos, scale=0.9)
            pos = dict(zip(meso_graph, pos))
            pos_df = pd.DataFrame(pos).sort_values(0, axis=1)
            pos_df.columns = range(nb_blocks)
            pos = pos_df.to_dict('list')
            nx.draw_networkx_nodes(meso_graph, pos,
                                   node_size=np.array(list(node_size))
                                   * scale**4 / 1.5, alpha=0.75,
                                   node_color='purple', margins=(0.1, 0.1))
            nx.draw_networkx_labels(meso_graph, pos, labels=node_labels,
                                    font_size=node_label_size,
                                    font_family='serif')
            for edge in meso_graph.edges(data=True):
                w = edge[2]['width']
                a = edge[2]['arrowsize']
                nx.draw_networkx_edges(meso_graph, pos,
                                       edgelist=[(edge[0], edge[1])],
                                       arrowsize=a * scale**2 / 2,
                                       width=w/nb_connect_mat.sum() * scale,
                                       node_size=np.array(list(node_size))
                                       * scale**4 / 2.5)
        # Undirected case:
        else:
            meso_graph = nx.Graph()
            for (i, j) in clust_pairs:
                clust_pairs_tuple.append((i, j))
                # Block of the adjacency matrix corresponding to connections
                # from fold changes in cluster i to those in j:
                clusters_i_j_adj_mat = self.adj_mat[clusters ==
                                                    i, :][:, clusters == j]
                nb_connect_mat[i, j] = clusters_i_j_adj_mat.sum()
                meso_graph.add_edge(i, j,
                                    width=nb_connect_mat[i, j] * obj_scale**2)
                # Annotation:
                if nb_connect_mat[i, j] != 0:
                    edge_labels[(i, j)] = f"{nb_connect_mat[i, j]}"

            node_labels = {n: self.var_names[centroids[n]] for n in meso_graph}
            sns.set_style("white")

            # Plot the graph:
            plt.figure(figsize=figsize)
            theta = np.linspace(0, 1, nb_blocks + 1)[:-1] * 2 * np.pi
            theta = theta.astype(np.float32)
            pos = np.column_stack([np.cos(theta), np.sin(theta),
                                   np.zeros((nb_blocks, 0))])
            pos = nx.rescale_layout(pos, scale=0.9)
            pos = dict(zip(meso_graph, pos))
            pos_df = pd.DataFrame(pos).sort_values(0, axis=1)
            pos_df.columns = range(nb_blocks)
            pos = pos_df.to_dict('list')
            nx.draw_networkx_nodes(meso_graph, pos,
                                   node_size=np.array(list(node_size))
                                   * scale**4 / 1.5, alpha=0.75,
                                   node_color='purple', margins=(0.1, 0.1))
            nx.draw_networkx_labels(meso_graph, pos, labels=node_labels,
                                    font_size=node_label_size,
                                    font_family='serif')
            for edge in meso_graph.edges(data=True):
                w = edge[2]['width']
                nx.draw_networkx_edges(meso_graph, pos,
                                       edgelist=[(edge[0], edge[1])],
                                       width=w/nb_connect_mat.sum() * scale,
                                       node_size=np.array(list(node_size))
                                       * scale**4 / 2.5)
        nx.draw_networkx_edge_labels(meso_graph, pos, edge_labels=edge_labels,
                                     font_size=15 * obj_scale**2,
                                     font_family='serif',
                                     bbox={'fc': 'w', 'ec': 'k'})
        plt.show()

    def graph_analysis(self, clusters, nb_top=10):
        """
        Performs a series of graph analyses of the fold changes network, in
        particular: identifies among the entities nb_top top hits, authorities,
        nodes with respect to pagerank, degree and betweenness centrality. It
        also plots a figure displaying degree distribution of the nodes.

        Parameters
        ----------
        clusters : ndarray
            1D array of length nb_var containing integers indicating clusters
            to which the fold changes are assigned.
        nb_top : int, optional
            Number of top elements to include. The default is 10.

        Returns
        -------
        graph_analysis : DataFrame
            2D DataFrame containing the names of entities that appeared in at
            least one of the considered tops as rows, and the following
            information as columns: cluster (number), all of the considered
            tops (1 if among the corresponding top and 0 otherwise), and total
            sum of all columns except the cluster one. Ordered so that the
            entities with the highest total score are at the top.

        """
        vn_dict = {i: self.var_names[i] for i in range(self.nb_var)}
        graph = nx.relabel_nodes(nx.DiGraph(self.adj_mat), vn_dict)
        # General information:
        nx.info(graph)

        # Hubs and authorities:
        (hubs, auth) = nx.hits(graph)
        top_hubs = pd.Series(hubs).sort_values(
            ascending=False).iloc[:nb_top].index
        top_auth = pd.Series(auth).sort_values(
            ascending=False).iloc[:nb_top].index

        # PageRank:
        pagerank = nx.pagerank(graph)
        top_pagerank = pd.Series(pagerank).sort_values(
            ascending=False).iloc[:nb_top].index

        # Degree analysis (code from https://networkx.org/documentation/stable/auto_examples/drawing/plot_degree.html):
        mpl.style.use('seaborn')
        mpl.rcParams['font.family'] = 'serif'

        degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)
        top_degree = pd.DataFrame(graph.degree()).set_index(
            0).sort_values(1).index[:nb_top]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle('Degree distribution')
        axs[0].plot(degree_sequence, "b-", marker="o")
        axs[0].set_title("Degree Rank Plot")
        axs[0].set_ylabel("Degree")
        axs[0].set_xlabel("Rank")

        axs[1].bar(*np.unique(degree_sequence, return_counts=True))
        axs[1].set_title("Degree histogram")
        axs[1].set_xlabel("Degree")
        axs[1].set_ylabel("# of Nodes")

        fig.tight_layout()
        plt.show()

        # Betweenness centrality:
        bc = nx.betweenness_centrality(graph)
        top_bc = pd.Series(bc).sort_values(ascending=False).iloc[:nb_top].index

        # Summary:
        all_measure_indices = (top_hubs.union(top_auth).union(top_bc)
                               .union(top_degree).union(top_pagerank))
        graph_analysis = pd.DataFrame(np.zeros((len(all_measure_indices),
                                                6), dtype=int),
                                      index=all_measure_indices,
                                      columns=['Cluster', 'Top degree',
                                               'Top central', 'Top hub',
                                               'Top authority', 'Top rank'])
        graph_analysis['Cluster'] = clusters[np.intersect1d(self.var_names,
                                                            all_measure_indices,
                                                            return_indices=True)[1]]
        graph_analysis['Top degree'] = all_measure_indices.isin(
            top_degree).astype(int)
        graph_analysis['Top central'] = all_measure_indices.isin(
            top_bc).astype(int)
        graph_analysis['Top hub'] = all_measure_indices.isin(
            top_hubs).astype(int)
        graph_analysis['Top authority'] = all_measure_indices.isin(
            top_auth).astype(int)
        graph_analysis['Top pagerank'] = all_measure_indices.isin(
            top_pagerank).astype(int)
        graph_analysis['Total'] = graph_analysis.drop(
            columns='Cluster').sum(axis=1)
        graph_analysis.sort_values(by='Total', ascending=False, inplace=True)

        return graph_analysis

    def pathway_search(self, clusters):
        """
        Identifies all shortest paths between entities in the network of length
        3 and bigger, and presents them along with their scores with respect to
        criteria potentially relevant for hypothesis generation.

        Parameters
        ----------
        clusters : ndarray
            1D array of length nb_var containing integers indicating clusters
            to which the fold changes are assigned.

        Returns
        -------
        all_paths_dict : dict
            Dictionary with keys of type 'string' indicating the path length l,
            and the values are dataframes. Each raw of such dataframe
            corresponds to a path, the names of the nodes listed in the first
            l columns. There are three other columns: warp score (number of
            strictly positive warps in the path, i.e. number of predictive
            relationships), cluster score (number of times there is a change
            in cluster in the path), and total score (sum of the first two).
            Paths in the dataframe are ordered with respect to the total score
            (highest to lowest).

        """
        graph = nx.DiGraph(self.adj_mat)
        shortest_pairs = pd.DataFrame(dict(nx.all_pairs_shortest_path(graph)))
        shortest_pairs_length = pd.DataFrame(
            dict(nx.all_pairs_shortest_path_length(graph)))
        max_length = int(shortest_pairs_length.max().max())
        all_paths_dict = {}
        # Iterating on path lengths:
        for l in range(3, max_length+1):
            l_paths = (pd.melt(shortest_pairs.where(shortest_pairs_length == l))
                       .dropna()['value'])
            l_paths_df = pd.DataFrame.from_dict(dict(zip(l_paths.index,
                                                         l_paths.values)),
                                                orient='index')
            l_paths_df['Warp score'] = np.zeros(
                (l_paths_df.shape[0]), dtype=int)
            l_paths_df['Cluster score'] = np.zeros(
                (l_paths_df.shape[0]), dtype=int)
            # Iterate simultaneously on nodes of paths of length l:
            for c in range(l):
                l_paths_df['Warp score'] += self.optimal_warp_mat[l_paths_df[c],
                                                                  l_paths_df[c+1]]
                cl_1 = clusters[l_paths_df.iloc[:, c]].squeeze()
                cl_2 = clusters[l_paths_df.iloc[:, (c+1)]].squeeze()
                l_paths_df['Cluster score'] += (cl_1 != cl_2).astype(int)
            l_paths_df['Total score'] = (l_paths_df['Warp score']
                                         + l_paths_df['Cluster score'])
            l_paths_df.sort_values(
                'Total score', ascending=False, inplace=True)
            vn_dict = {i: self.var_names[i] for i in range(self.nb_var)}
            l_paths_df.iloc[:, :(l + 1)] = (l_paths_df.iloc[:, :(l + 1)]
                                            .replace(vn_dict))
            all_paths_dict[f'{l} paths'] = l_paths_df
        return all_paths_dict
