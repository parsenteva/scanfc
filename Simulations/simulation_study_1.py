"""
Script performing the first series of simulation studies.

@author: Polina Arsenteva
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scanfc.clustering import FoldChanges, Clustering
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
from simulation_functions import simulate_cluster_means_1
mpl.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'


def compute_big_sigma(fc_cov):
    """
    Transforms a 3D covariance tensor into a 2D version with vertically and
    horizontally stacked covariance matrices.

    Parameters
    ----------
    fc_cov : ndarray
        3D array of shape (nb_time_pts, nb_var, nb_var)
        containing data with `float` type, representing fold changes'
        nb_var x nb_var shaped covariance matrices for each time point.

    Returns
    -------
    big_sigma : ndarray
        2D array of shape (nb_var * nb_time_pts, nb_var * nb_time_pts)
        containing data with `float` type, representing fold changes'
        nb_time_pts x nb_time_pts shaped covariance matrices for each pair
        of entities stacked together.

    """
    dim1 = fc_cov.shape[0]
    dim2 = fc_cov.shape[1]
    dim = dim1 * dim2
    fc_var = np.diagonal(fc_cov, axis1=1, axis2=2)
    big_sigma = np.zeros((dim, dim))
    big_sigma[np.diag_indices(dim)] = np.ndarray.flatten(fc_var.T)
    index_pairs = np.array(list(itertools.combinations(range(dim2), 2)))
    for pair in index_pairs:
        big_sigma[dim1*pair[0]:dim1*(pair[0]+1),
                  dim1*pair[1]:dim1*(pair[1]+1)][np.diag_indices(dim1)] = fc_cov[:, pair[0], pair[1]]
        big_sigma[dim1*pair[1]:dim1*(pair[1]+1),
                  dim1*pair[0]:dim1*(pair[0]+1)][np.diag_indices(dim1)] = fc_cov[:, pair[0], pair[1]]
    return big_sigma


def perform_simulation_study(k, sim_clusters, sim_means, sim_cov, time_points,
                             nb_sim_rep, random_gen=None, nb_cl_rep=50):
    """
    Performs a simulation study focusing on distances and clustering algorithms.

    Parameters
    ----------
    k : int
        Number of simulated clusters.
    sim_clusters : ndarray
        1D array of length nb_var containing integers indicating clusters
        to which the simulated fold changes are assigned
        (in the original order).
    sim_means : ndarray
        2D array of shape (nb_time_points, nb_var) containing data
        with `float` type, representing simulated fold changes' means for
        each entity and each time point.
    sim_cov : ndarray
        3D array of shape (nb_time_pts, nb_var, nb_var) containing
        data with `float` type, representing simulated fold changes'
        nb_var x nb_var shaped covariance matrices for each time point.
        Time-wise cross-covariances are assumes to be 0 due to experimental
        design. In case of Hellinger distance, can also be 4-dimensional
        (natural form): (nb_time_pts, nb_time_pts, nb_var, nb_var).
    time_points : array-like
        1D array-like containing data with `float` type, representing time
        points at which fold changes were simulated.
    nb_sim_rep : int
        Number of times each of the considered approaches is repeated, based
        on these repetitions the mean values and standard deviation are
        calculated.
    random_gen : RandomState instance or None
        Random number generator, used to reproduce results. If None (default),
        the generator is the RandomState instance used by `np.random`.
        If RandomState instance, random_gen is the actual random
        number generator.
    nb_cl_rep : int, optional
        Number of random initialization attempts. The default is 50.

    Returns
    -------
    sim_results_cost : ndarray
        2D array of shape (nb_sim_rep, 6), where 6 correspond to the number of
        considered approaches, containing the comparable cost values for each
        repetition and each approach.
    sim_results_ars : ndarray
        2D array of shape (nb_sim_rep, 6), where 6 correspond to the number of
        considered approaches, containing adjusted rand scores for each
        repetition and each approach.
    sim_results_vm : ndarray
        2D array of shape (nb_sim_rep, 6), where 6 correspond to the number of
        considered approaches, containing V-measure values for each
        repetition and each approach.

    """
    sim_fc = FoldChanges(means=sim_means, cov=sim_cov, time_points=time_points)
    cl_kmed_sim = Clustering(fold_changes=sim_fc, dist='d2hat',
                             random_gen=random_gen)
    cl_wass_sim = Clustering(fold_changes=sim_fc, dist='wasserstein',
                             random_gen=random_gen)
    cl_hell_sim = Clustering(fold_changes=sim_fc, dist='hellinger',
                             random_gen=random_gen)

    sim_results_ars = np.zeros((nb_sim_rep, 6))
    sim_results_vm = np.zeros((nb_sim_rep, 6))
    sim_results_cost = np.zeros((nb_sim_rep, 6))

    alg = 'k-means-like'
    for i in range(nb_sim_rep):
        print(f'iter {i+1}/{nb_sim_rep}')
        # d2hat K-medoid clustering of simulated data:
        (k_clusters,
         k_centroids,
         k_cost) = cl_kmed_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                             disp_plot=False, algorithm=alg)
        sim_results_cost[i, 0] = cl_kmed_sim.calculate_comparable_cost(k,
                                                                       k_clusters)
        sim_results_ars[i, 0] = adjusted_rand_score(sim_clusters, k_clusters)
        sim_results_vm[i, 0] = v_measure_score(sim_clusters, k_clusters)
        # K-medoids Wasserstein distance based clustering:
        (k_clusters_w_med,
         k_centroids_w_med,
         k_cost_w_med) = cl_wass_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                                     disp_plot=False, algorithm=alg)
        sim_results_cost[i, 1] = cl_wass_sim.calculate_comparable_cost(k,
                                                                         k_clusters_w_med)
        sim_results_ars[i, 1] = adjusted_rand_score(sim_clusters, k_clusters_w_med)
        sim_results_vm[i, 1] = v_measure_score(sim_clusters, k_clusters_w_med)
        
        # K-medoid clustering of simulated data (Hellinger distance):
        (k_clusters_hell,
         k_centroids_hell,
         k_cost_hell) = cl_hell_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                                  disp_plot=False,
                                                  algorithm=alg)
        sim_results_cost[i, 2] = cl_hell_sim.calculate_comparable_cost(k,
                                                                       k_clusters_hell)
        sim_results_ars[i, 2] = adjusted_rand_score(sim_clusters, k_clusters_hell)
        sim_results_vm[i, 2] = v_measure_score(sim_clusters, k_clusters_hell)
        # d2hat hierarchical clustering:
        (k_clusters_hier_d,
         k_centroids_hier_d) = cl_kmed_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                                         disp_plot=False,
                                                         method='hierarchical')
        sim_results_cost[i, 3] = cl_kmed_sim.calculate_comparable_cost(k,
                                                                       k_clusters_hier_d)
        sim_results_ars[i, 3] = adjusted_rand_score(sim_clusters, k_clusters_hier_d)
        sim_results_vm[i, 3] = v_measure_score(sim_clusters, k_clusters_hier_d)
        # Wasserstein hierarchical clustering:
        (k_clusters_hier_w,
         k_centroids_hier_w) = cl_wass_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                                         disp_plot=False,
                                                         method='hierarchical')
        sim_results_cost[i, 4] = cl_wass_sim.calculate_comparable_cost(k,
                                                                       k_clusters_hier_w)
        sim_results_ars[i, 4] = adjusted_rand_score(sim_clusters, k_clusters_hier_w)
        sim_results_vm[i, 4] = v_measure_score(sim_clusters, k_clusters_hier_w)
        # Hellinger hierarchical clustering:
        (k_clusters_hier_h,
         k_centroids_hier_h) = cl_hell_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                                         disp_plot=False,
                                                         method='hierarchical')
        sim_results_cost[i, 5] = cl_hell_sim.calculate_comparable_cost(k,
                                                                       k_clusters_hier_h)
        sim_results_ars[i, 5] = adjusted_rand_score(
            sim_clusters, k_clusters_hier_h)
        sim_results_vm[i, 5] = v_measure_score(sim_clusters, k_clusters_hier_h)

    print(
        f'd2hat k-medoids: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 0], axis=0)}, std {np.std(sim_results_ars[:, 0], axis=0)}')
    print(
        f'  V-measure score: mean {np.mean(sim_results_vm[:, 0], axis=0)}, std {np.std(sim_results_vm[:, 0], axis=0)}')
    print(
        f'  Cost: mean {np.mean(sim_results_cost[:, 0], axis=0)}, std {np.std(sim_results_cost[:, 0], axis=0)}')

    print(
        f'Wass k-medoids: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 1], axis=0)}, std {np.std(sim_results_ars[:, 1], axis=0)}')
    print(
        f'  V-measure score: mean {np.mean(sim_results_vm[:, 1], axis=0)}, std {np.std(sim_results_vm[:, 1], axis=0)}')
    print(
        f'  Cost: mean {np.mean(sim_results_cost[:, 1], axis=0)}, std {np.std(sim_results_cost[:, 1], axis=0)}')

    print(
        f'Hellinger k-medoids: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 2], axis=0)}, std {np.std(sim_results_ars[:, 2], axis=0)}')
    print(
        f'  V-measure score: mean {np.mean(sim_results_vm[:, 2], axis=0)}, std {np.std(sim_results_vm[:, 2], axis=0)}')
    print(
        f'  Cost: mean {np.mean(sim_results_cost[:, 2], axis=0)}, std {np.std(sim_results_cost[:, 2], axis=0)}')

    print(
        f'd2hat hierarchical: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 3], axis=0)}, std {np.std(sim_results_ars[:, 3], axis=0)}')
    print(
        f'  V-measure score: mean {np.mean(sim_results_vm[:, 3], axis=0)}, std {np.std(sim_results_vm[:, 3], axis=0)}')
    print(
        f'  Cost: mean {np.mean(sim_results_cost[:, 3], axis=0)}, std {np.std(sim_results_cost[:, 3], axis=0)}')
    
    print(
        f'Wass hierarchical: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 4], axis=0)}, std {np.std(sim_results_ars[:, 4], axis=0)}')
    print(
        f'  V-measure score: mean {np.mean(sim_results_vm[:, 4], axis=0)}, std {np.std(sim_results_vm[:, 4], axis=0)}')
    print(
        f'  Cost: mean {np.mean(sim_results_cost[:, 4], axis=0)}, std {np.std(sim_results_cost[:, 4], axis=0)}')
    
    print(
        f'Hellinger hierarchical: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 5], axis=0)}, std {np.std(sim_results_ars[:, 5], axis=0)}')
    print(
        f'  V-measure score: mean {np.mean(sim_results_vm[:, 5], axis=0)}, std {np.std(sim_results_vm[:, 5], axis=0)}')
    print(
        f'  Cost: mean {np.mean(sim_results_cost[:, 5], axis=0)}, std {np.std(sim_results_cost[:, 5], axis=0)}')
    

    return sim_results_cost, sim_results_ars, sim_results_vm


rand_seed = 597
random_gen = np.random.RandomState(rand_seed)

nb_var = 300
time_points = np.array([0.5,  1.,  2.,  3.,  4.,  7., 14., 21.])
nb_time_p = len(time_points)

######################## Distances & Algorithms Study #########################

############## Simulation of 4 clusters ##############
k = 4
sim_clusters_0 = random_gen.choice(range(k), nb_var)
random_gen.shuffle(sim_clusters_0)


# Simulating means of 4 clusters:
plot_time_points = np.linspace(0, 21, 43)
time_p_ind = np.array((1, 2, 4, 6, 8, 14, 28, 42))
sim_means_0 = np.zeros((nb_time_p, nb_var))
fig, axs = plt.subplots(1, k, figsize=(20, 8), sharey=False)
for i, cl in enumerate(range(k)):
    cluster_i = np.argwhere(sim_clusters_0 == cl)
    sim_means_plot_0 = simulate_cluster_means_1(cluster_i.size, plot_time_points,
                                              cl+1, random_gen=random_gen)
    axs[i].plot(plot_time_points, sim_means_plot_0)
    sim_means_0[:, np.squeeze(cluster_i)] = sim_means_plot_0[time_p_ind, :]
fig.suptitle('Study 1: functional representation of simulated means',
             fontsize=16)
plt.tight_layout()
plt.show()


####### Independent case #######
# Simulating variances of k clusters:
sim_cov_0 = np.zeros((nb_time_p, nb_var, nb_var))
sim_var_0 = np.repeat(np.abs(random_gen.randn(nb_var)*2),
                      nb_time_p).reshape((nb_var, nb_time_p)).T
sim_cov_0[:, np.arange(300), np.arange(300)] = sim_var_0

nb_sim_rep = 10


print('Study 1: d2hat vs Wasserstein vs Hellinger and k-medoids vs hierarchical')
print('4 clusters')
print('Independent case')
(sim_0_results_cost,
 sim_0_results_ars,
 sim_0_results_vm) = perform_simulation_study(k, sim_clusters_0, sim_means_0,
                                              sim_cov_0, time_points,
                                              nb_sim_rep, random_gen=random_gen)

########################## Simulation of 2 clusters ###########################
k = 2
sim_clusters = np.concatenate((np.zeros((nb_var//2)), np.ones((nb_var//2))))
random_gen.shuffle(sim_clusters)

# Simulating means of 2 clusters:
sim_means = np.zeros((nb_time_p, nb_var))
fig, axs = plt.subplots(1, k, figsize=(20, 8), sharey=False)
for i, cl in enumerate(range(k)):
    cluster_i = np.argwhere(sim_clusters == cl)
    sim_means_plot = simulate_cluster_means_1(cluster_i.size, plot_time_points,
                                            cl+1, random_gen=random_gen)
    axs[i].plot(plot_time_points, sim_means_plot)
    sim_means[:, np.squeeze(cluster_i)] = sim_means_plot[time_p_ind, :]
fig.suptitle('Study 1: functional representation of simulated means',
             fontsize=16)
plt.tight_layout()
plt.show()

####### Independent case #######
# Simulating variances of k clusters:
sim_cov_1 = np.zeros((nb_time_p, nb_var, nb_var))
sim_var_1 = np.repeat(np.abs(random_gen.randn(nb_var)*2),
                      nb_time_p).reshape((nb_var, nb_time_p)).T
sim_cov_1[:, np.arange(300), np.arange(300)] = sim_var_1

print('2 clusters')
print('Independent case')
(sim_1_results_cost,
 sim_1_results_ars,
 sim_1_results_vm) = perform_simulation_study(k, sim_clusters, sim_means,
                                              sim_cov_1, time_points,
                                              nb_sim_rep, random_gen=random_gen)

####### Block-dependent case #######
# Simulating covariances of k clusters:
# Low covariances
sim_cov_21 = np.zeros((nb_time_p, nb_var, nb_var))
sim_cov_21[:, np.arange(300), np.arange(300)] = sim_var_1
theta = 1
for i in range(k):
    cluster_i = np.argwhere(sim_clusters == i)
    cluster_i_pairs = np.array(
        list(itertools.combinations(np.squeeze(cluster_i), 2)))
    sim_cov_21[:, cluster_i_pairs[:, 0],
               cluster_i_pairs[:, 1]] = random_gen.uniform(0, 1 * theta,
                                                           cluster_i_pairs.shape[0])
    sim_cov_21[:, cluster_i_pairs[:, 1],
               cluster_i_pairs[:, 0]] = sim_cov_21[:, cluster_i_pairs[:, 0],
                                                   cluster_i_pairs[:, 1]].copy()

big_sigma_sqrt = compute_big_sigma(sim_cov_21)
big_sigma = big_sigma_sqrt @ big_sigma_sqrt.T
big_sigma_reshaped = big_sigma.reshape(
    (nb_var, nb_time_p, nb_var, nb_time_p)).transpose((1, 3, 0, 2))
big_sigma_3d = np.diagonal(big_sigma_reshaped, axis1=0, axis2=1).T
index_pairs = np.array(list(itertools.combinations(range(nb_var), 2)))
big_sigma_scaled = big_sigma_3d / np.max(big_sigma_3d[:, index_pairs[:, 0],
                                                      index_pairs[:, 1]])

print('Block-dependent case (low cov)')
(sim_21_results_cost,
 sim_21_results_ars,
 sim_21_results_vm) = perform_simulation_study(k, sim_clusters, sim_means,
                                               big_sigma_scaled, time_points,
                                               nb_sim_rep, random_gen=random_gen)

####### Block-dependent case #######
# Simulating covariances of k clusters:
# High covariances
sim_cov_22 = np.zeros((nb_time_p, nb_var, nb_var))
sim_cov_22[:, np.arange(300), np.arange(300)] = sim_var_1
theta = 1
for i in range(k):
    cluster_i = np.argwhere(sim_clusters == i)
    cluster_i_pairs = np.array(
        list(itertools.combinations(np.squeeze(cluster_i), 2)))
    sim_cov_22[:, cluster_i_pairs[:, 0],
               cluster_i_pairs[:, 1]] = random_gen.uniform(0, 1 * theta,
                                                           cluster_i_pairs.shape[0])
    sim_cov_22[:, cluster_i_pairs[:, 1],
               cluster_i_pairs[:, 0]] = sim_cov_22[:, cluster_i_pairs[:, 0],
                                                   cluster_i_pairs[:, 1]].copy()

big_sigma_sqrt = compute_big_sigma(sim_cov_22)
big_sigma = big_sigma_sqrt @ big_sigma_sqrt.T
big_sigma_reshaped = (big_sigma.reshape((nb_var, nb_time_p,
                                        nb_var, nb_time_p))
                      .transpose((1, 3, 0, 2)))
big_sigma_scaled = big_sigma_3d / 20


print('Block-dependent case (high cov)')
(sim_22_results_cost,
 sim_22_results_ars,
 sim_22_results_vm) = perform_simulation_study(k, sim_clusters, sim_means,
                                               big_sigma_scaled, time_points,
                                               nb_sim_rep, random_gen=random_gen)

####### Case: Positive vs. Negative #######
# Simulating covariances of k clusters:
# Low covariances
size_cluster_1 = np.count_nonzero(sim_clusters == 0)
size_cluster_2 = np.count_nonzero(sim_clusters == 1)
block_sigma_1 = random_gen.uniform(0, 1, (size_cluster_1, size_cluster_1))
block_sigma_2 = random_gen.uniform(-1, 0, (size_cluster_2, size_cluster_2))
sigmas_concat = np.concatenate((block_sigma_1, block_sigma_2), axis=0)
sigma = (sigmas_concat @ sigmas_concat.T) / 100
cluster_0 = np.squeeze(np.argwhere(sim_clusters == 0))
cluster_1 = np.squeeze(np.argwhere(sim_clusters == 1))
new_order = []
count_cluster_0 = 0
count_cluster_1 = 0
for i in range(nb_var):
    new_order.append(count_cluster_0 * (sim_clusters[i] == 0)
                     + (150 + count_cluster_1) * (sim_clusters[i] == 1))
    count_cluster_0 += 1 * (sim_clusters[i] == 0)
    count_cluster_1 += 1 * (sim_clusters[i] == 1)
sigma_reordered = sigma[tuple(new_order), :][:, tuple(new_order)]
sigma_3d = (sigma_reordered.repeat(nb_time_p).reshape((nb_var, nb_var, nb_time_p))
            .transpose((2, 0, 1)))
sigma_scaled = sigma_3d*1


print('Positive vs. Negative case (low cov)')
(sim_31_results_cost,
 sim_31_results_ars,
 sim_31_results_vm) = perform_simulation_study(k, sim_clusters, sim_means,
                                               sigma_scaled, time_points,
                                               nb_sim_rep, random_gen=random_gen)

####### Case: Positive vs. Negative #######
# Simulating covariances of k clusters:
# High covariances
size_cluster_1 = np.count_nonzero(sim_clusters == 0)
size_cluster_2 = np.count_nonzero(sim_clusters == 1)
block_sigma_1 = random_gen.uniform(0, 1, (size_cluster_1, size_cluster_1))
block_sigma_2 = random_gen.uniform(-1, 0, (size_cluster_2, size_cluster_2))
sigmas_concat = np.concatenate((block_sigma_1, block_sigma_2), axis=0)
sigma = (sigmas_concat @ sigmas_concat.T) / 100
cluster_0 = np.squeeze(np.argwhere(sim_clusters == 0))
cluster_1 = np.squeeze(np.argwhere(sim_clusters == 1))
new_order = []
count_cluster_0 = 0
count_cluster_1 = 0
for i in range(nb_var):
    new_order.append(count_cluster_0 * (sim_clusters[i] == 0) + (
        150 + count_cluster_1) * (sim_clusters[i] == 1))
    count_cluster_0 += 1 * (sim_clusters[i] == 0)
    count_cluster_1 += 1 * (sim_clusters[i] == 1)
sigma_reordered = sigma[tuple(new_order), :][:, tuple(new_order)]
sigma_3d = sigma_reordered.repeat(nb_time_p).reshape(
    (nb_var, nb_var, nb_time_p)).transpose((2, 0, 1))
sigma_scaled = sigma_3d*2


print('Positive vs. Negative case (high cov)')
(sim_32_results_cost,
 sim_32_results_ars,
 sim_32_results_vm) = perform_simulation_study(k, sim_clusters, sim_means,
                                               sigma_scaled, time_points,
                                               nb_sim_rep, random_gen=random_gen)


############################### Results summary ###############################
ars_table = [np.round(np.mean(sim_0_results_ars, axis=0), 2),
             np.round(np.mean(sim_1_results_ars, axis=0), 2),
             np.round(np.mean(sim_21_results_ars, axis=0), 2),
             np.round(np.mean(sim_22_results_ars, axis=0), 2),
             np.round(np.mean(sim_31_results_ars, axis=0), 2),
             np.round(np.mean(sim_32_results_ars, axis=0), 2)]
vm_table = [np.round(np.mean(sim_0_results_vm, axis=0), 2),
            np.round(np.mean(sim_1_results_vm, axis=0), 2),
            np.round(np.mean(sim_21_results_vm, axis=0), 2),
            np.round(np.mean(sim_22_results_vm, axis=0), 2),
            np.round(np.mean(sim_31_results_vm, axis=0), 2),
            np.round(np.mean(sim_32_results_vm, axis=0), 2)]

fig, axs = plt.subplots(1, 2, figsize=(30, 7))
pos_0 = axs[0].imshow(ars_table, aspect='auto', cmap='RdYlGn')
pos_1 = axs[1].imshow(vm_table, aspect='auto', cmap='RdYlGn')

# add the values
for (i, j), value in np.ndenumerate(ars_table):
    axs[0].text(j, i, "%.3f" % value, va='center', ha='center', fontsize=20)
for (i, j), value in np.ndenumerate(vm_table):
    axs[1].text(j, i, "%.3f" % value, va='center', ha='center', fontsize=20)

axs[0].axis('off')
axs[0].grid(False)
axs[0].text(0, -0.7, 'd2hat \n k-medoids',
            va='center', ha='center', fontsize=15)
axs[0].text(1, -0.7, 'Wasserstein \n k-medoids',
            va='center', ha='center', fontsize=15)
axs[0].text(2, -0.7, 'Hellinger \n k-medoids',
            va='center', ha='center', fontsize=15)
axs[0].text(3, -0.7, 'd2hat \n hierarchical',
            va='center', ha='center', fontsize=15)
axs[0].text(4, -0.7, 'Wasserstein \n hierarchical',
            va='center', ha='center', fontsize=15)
axs[0].text(5, -0.7, 'Hellinger \n hierarchical',
            va='center', ha='center', fontsize=15)
axs[0].text(2.25, -1.1, 'ARI', fontsize=25)
middle_coord = 6.1
axs[0].text(middle_coord, 0, '4 clusters:\nIndependent', ha='center', fontsize=13)
axs[0].text(middle_coord, 1.1, '2 clusters:\nIndependent', ha='center', fontsize=13)
axs[0].text(middle_coord, 2.2, '2 clusters:\nBlock-dependent\n(low cov.)',
            ha='center', fontsize=13)
axs[0].text(middle_coord, 3.2, '2 clusters:\nBlock-dependent\n(high cov.)',
            ha='center', fontsize=13)
axs[0].text(middle_coord, 4.2, '2 clusters:\n +/- \n(low cov.)', ha='center', fontsize=13)
axs[0].text(middle_coord, 5.2, '2 clusters:\n +/- \n(high cov.)',
            ha='center', fontsize=13)
axs[1].axis('off')
axs[1].grid(False)
axs[1].text(0, -0.7, 'd2hat \n k-medoids',
            va='center', ha='center', fontsize=15)
axs[1].text(1, -0.7, 'Wasserstein \n k-medoids',
            va='center', ha='center', fontsize=15)
axs[1].text(2, -0.7, 'Hellinger \n k-medoids',
            va='center', ha='center', fontsize=15)
axs[1].text(3, -0.7, 'd2hat \n hierarchical',
            va='center', ha='center', fontsize=15)
axs[1].text(4, -0.7, 'Wasserstein \n hierarchical',
            va='center', ha='center', fontsize=15)
axs[1].text(5, -0.7, 'Hellinger \n hierarchical',
            va='center', ha='center', fontsize=15)

axs[1].text(1.75, -1.1, 'V-measure', fontsize=25)
fig.colorbar(pos_0, ax=axs, anchor=(-0.25, 0.5))
fig.savefig("sim_cc_rev.pdf")
plt.show()
