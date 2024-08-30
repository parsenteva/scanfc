"""
ScanFC : Statistical framework for Clustering with Alignment and
    Network inference of Fold Changes.

@author: Polina Arsenteva

"""
import itertools
import warnings
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
for style in plt.style.available:
    if 'seaborn' in style:
        mpl.style.use(style)
        break


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

class Clustering():
    """
    A class containing tools for clustering fold changes.

    Attributes
    ----------
    fold_changes : FoldChanges instance or None
        None if dist_mat is given by the user for clustering. Otherwise, is
        equal to the corresponding parameter, which has to be non_none, and
        corresponds to the fold changes considered for clustering.
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
        Initializes centroids (medoids) for k clusters.
    assign_clusters(centroids)
        Assigns all fold changes to one of the k clusters based on their
        distances to centroids (medoids).
    update_centroids(k, clusters, old_centroids, algorithm='k-means-like')
        Recalculates centroids based on the current cluster configuration.
    hierarchical_centroids(k, clusters)
        Chooses centroids among the fold changes in clusters after clustering.
        Used for non-centroid based clustering methods, such as hierarchical
        clustering.
    calculate_total_cost(centroids, clusters)
        Calculates total cost for all clusters, defined as the sum of distances
        between the fold changes and their centroids with respect to the
        distance matrix. Used in k-medoids clustering as a selection criterion.
    calculate_comparable_cost(k, clusters)
        Calculates total comparable cost for all clusters, defined as the sum
        of distances between all fold change pairs in each cluster with respect
        to the distance matrix. Used to compare clustering performed with
        different methods (distance matrix should be the same).
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
    plot_clusters(k, clusters, centroids, warps=None, nb_cols=4, 
                  nb_rows=None, figsize=None)
        Produces a figure with k subplots (or 2 figures if warps are provided),
        each containing plots of the fold changes' means in the corresponding
        cluster. In the case with time warping, produces a figure with
        unaligned (original) and a figure with aligned (with respect to their
        centroids) fold changes.

    """

    def __init__(self, dist_mat=None, optimal_warp_mat=None, fold_changes=None, 
                 dist='d2hat', time_warp=False, max_warp_step=1, sign_pen=False,
                 pen_param=1, random_gen=None):
        """
        Parameters
        ----------
        dist_mat : ndarray or None
            Distance matrix, 2D array of shape with both dimensions
            equal to the number of fold changes (entities). Either 'dist_mat' 
            or 'fold_changes' has to be non-None, with 'dist_mat' having 
            priority for the clustering, in which case the latter is performed
            directly on the distance matrix.
        optimal_warp_mat : ndarray or None
            Optimal Warp matrix, 2D array with both dimensions equal to the
            number of fold changes (entities). If provided, time_warp is set to
            True and alignment is integrated into different parts of the pipeline.
        fold_changes : FoldChanges instance or None
            If provided, and if dist_mat is None, then dist_mat and other 
            attributes are constructed based on the fold changes.
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
        self.fold_changes = fold_changes
        if dist_mat is not None:
            self.dist_mat = dist_mat
            self.nb_var = self.dist_mat.shape[0]
            self.index_pairs = np.array(
                list(itertools.combinations(range(self.nb_var), 2)))
            self.distances = self.dist_mat[self.index_pairs[:, 0],
                                           self.index_pairs[:, 1]]
            if optimal_warp_mat is not None:
                self.time_warp = True
                self.optimal_warp_mat = optimal_warp_mat
            else:
                self.time_warp = False
        else:
            assert_str = "Either 'dist_mat' or a 'fold_changes' has to be non-None."
            assert self.fold_changes is not None, assert_str
            self.nb_var = self.fold_changes.nb_var
            self.dist = dist
            self.time_warp = time_warp
            if sign_pen:
                self.sign_pen = sign_pen
                self.pen_param = pen_param
            # With alignment:
            if time_warp:
                assert max_warp_step >= 0
                self.max_warp_step = max_warp_step
                if self.fold_changes.means is not None:
                    (self.index_pairs,
                     warped_distances) = (self.fold_changes
                                          .compute_warped_distance_pairs(max_warp_step=max_warp_step,
                                                                         sign_pen=sign_pen,
                                                                         pen_param=pen_param))
                    self.distances = np.min(warped_distances, axis=0)
            # Without alignment:
            else:
                if self.fold_changes.means is not None:
                    (self.index_pairs,
                     self.distances) = (self.fold_changes
                                        .compute_distance_pairs(dist=self.dist,
                                                                sign_pen=sign_pen,
                                                                pen_param=pen_param))
            if self.fold_changes.means is not None:
                if self.dist in ('d2hat', 'hellinger'):
                    if time_warp:
                        (self.dist_mat,
                         self.optimal_warp_mat) = (self.fold_changes
                                                   .compute_warped_dist_mat(self.index_pairs,
                                                                            warped_distances))
                    else:
                        self.dist_mat = (self.fold_changes
                                         .compute_dist_mat(self.index_pairs,
                                                           self.distances))
                if self.dist == 'wasserstein':
                    fc_var = np.diagonal(self.fold_changes.cov, axis1=1, axis2=2)
                    id_tensor = (np.repeat(np.identity(self.fold_changes.nb_time_pts),
                                           self.nb_var, axis=1)
                                 .reshape((self.fold_changes.nb_time_pts, self.fold_changes.nb_time_pts,
                                           self.nb_var)))
                    M1 = np.einsum('ijk,ik->ijk', id_tensor, fc_var)
                    self.dist_mat = (self.fold_changes
                                     .compute_cross_distances(self.fold_changes.means, M1)[0])
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

    def assign_clusters(self, centroids):
        """
        Assigns all fold changes to one of the k clusters based on their
        distances to centroids (medoids).

        Parameters
        ----------
        centroids : ndarray
            1D array of length k containing indices in range (0, nb_var) of
            the fold changes that act as current centroids (medoids). Used for
            clusters assignment only if method=='k-medoids'.
        Returns
        -------
        clusters : ndarray
            1D array of length nb_var containing integers in range (0, k)
            indicating clusters to which the fold changes are assigned.

        """
        centroids_int = centroids.astype(int)
        all_vs_centroids = np.copy(self.dist_mat[centroids_int, :])
        clusters = np.argmin(all_vs_centroids, axis=0)
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
            centroids : ndarray
                1D array of length k containing indices in range (0, nb_var) of
                the fold changes that have been chosen as centroids.
            total_cost : float
                Value of the final total clustering cost with respect to the
                metric associated with the chosen clustering method.
                Returned as the last element of the list if method=='k-medoids'
                (in other cases absent since the cost isn't assessed during 
                clustering and should be calculated separately if needed).

        """
        mpl.rcParams['font.family'] = 'serif'
        # Initialize clusters with k-means++ (used for d2hat k-medoids):
        centroids = self.init_centroids(k)

        # Iterate assign clusters & recalculate centroids until
        # criterion is satisfied :
        flag = False
        i = 0
        if method == 'k-medoids':
            while not flag:
                clusters = self.assign_clusters(centroids)
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
        if method == 'hierarchical':
            h_clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed',
                                                   linkage='complete')
            clusters = h_clustering.fit_predict(self.dist_mat)
            centroids = self.hierarchical_centroids(k, clusters)
            return clusters, centroids
        if method == 'umap':
            warnings.filterwarnings('ignore', '.*precomputed metric.*')
            import umap
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
        k : int or list of ints
            Number of clusters.
        nb_rep : int, optional
            Number of random initialization attempts (k-means clustering
            initializations performed on the UMAP projection if
            method=='umap'). The default is 100.
        method : str, optional
            Main approach to clustering, options include:
                - 'k-medoids' (default, coupled with d2hat distance or
                Hellinger distance),
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
        If time_warp is True, a new element 'warps' (or 'all_warps' if
        if different numbers of clusters are considered) is added to the list.
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
                                                    metric='precomputed',
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

    def plot_clusters(self, k, clusters, centroids, warps=None, nb_cols=4,
                      nb_rows=None, figsize=None):
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
            If ndarray, 1D array of length k containing indices in range
            (0, nb_var) of the fold changes that act as centroids.
            If a dictionary, the keys are numbers of clusters considered,
            and for each such number the value is the latter array.
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
        assert_str = "Define 'fold_changes' as a FoldChanges instance to use this function."
        assert self.fold_changes is not None, assert_str
        
        sns.set_style("darkgrid")
        mpl.rcParams['font.family'] = 'serif'
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
        if self.fold_changes.time_points is not None:
            time_points = self.fold_changes.time_points
        else:
            time_points = range(self.fold_changes.nb_time_pts)
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
                                           self.fold_changes.means[:, cluster_j[f]],
                                           ':', color=c)
                else:
                    ax_j[col_to_plot].plot(time_points,
                                           self.fold_changes.means[:, cluster_j[f]],
                                           ':', color='grey')
            # the centroid is colored in red and thick
            ax_j[col_to_plot].plot(time_points, self.fold_changes.means[:, centroids_k[j]],
                                   '-', color='red', linewidth=2,
                                   label='Centroid')
            if self.fold_changes.var_names is not None:
                cluster_title = f'Centroid: {self.fold_changes.var_names[centroids_k[j]]}, {len(cluster_j)} members'
                ax_j[col_to_plot].set_title(cluster_title, color='red')
            else:
                cluster_title = f'{len(cluster_j)} members'
                ax_j[col_to_plot].set_title(cluster_title, color='red')
            if warps is not None:  # adding a legend specifying warp types
                lines = [Line2D([0], [0], color=c, linewidth=1,
                                linestyle=':') for c in warp_colors]
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
                                                    self.fold_changes.nb_time_pts-max(0, warp_type)))
                    plot_range = np.asarray(range(max(0, warp_type),
                                                  self.fold_changes.nb_time_pts+min(0, warp_type)))
                    c = warp_colors[np.max(warps_k) + warp_type]
                    ax_j[col_to_plot].plot(time_points[plot_range],
                                           self.fold_changes.means[warped_range,
                                                      cluster_j[f]],
                                           ':', color=c)
                ax_j[col_to_plot].plot(time_points, self.fold_changes.means[:, centroids_k[j]],
                                       '-', color='red', linewidth=2, label='Centroid')
                if self.fold_changes.var_names is not None:
                    cluster_title = f'Centroid: {self.fold_changes.var_names[centroids_k[j]]}, {len(cluster_j)} members'
                    ax_j[col_to_plot].set_title(cluster_title, color='red')
                # Adding a legend specifying warp types
                lines = [Line2D([0], [0], color=c, linewidth=1,
                                linestyle=':') for c in warp_colors]
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

