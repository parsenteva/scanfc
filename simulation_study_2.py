"""
Script performing the second series of simulation studies.

@author: Polina Arsenteva
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scanofc import Clustering, NetworkInference
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score
mpl.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

def simulate_cluster_means(size, time_points, func, random_gen=None):
    """
    Simulates means for a given cluster for the second series of simulation
    studies (with horizontal shifts).

    Parameters
    ----------
    size : int
        Number of members in the cluster.
    time_points : array-like
        1D array-like containing data with `float` type, representing time
        points for the simulated dataset.
    func : int (1, 2, 3 or 4)
        Indicates a simulation model for cluster means, different models
        generate different clusters. Each model is based on a polynomial
        (except 4 which is sine-based) with random coefficients.
    random_gen : RandomState instance or None, optional
        Random number generator, used to reproduce results. If None (default),
        the generator is the RandomState instance used by `np.random`.
        If RandomState instance, random_gen is the actual random
        number generator.

    Returns
    -------
    sim_clust_means_unaligned : ndarray
        2D array of shape (len(time_points), size) containing data
        with `float` type, representing unaligned fold changes' means for each
        simulated entity and each time point.
    sim_clust_means_aligned : ndarray
        2D array of shape (len(time_points), size) containing data
        with `float` type, representing aligned fold changes' means for each
        simulated entity and each time point.

    """
    nb_time_p = len(time_points)
    time_rep = np.repeat(time_points, size).reshape(nb_time_p, size)
    if random_gen is None:
        random_gen = np.random
    if func==1:
        # Model with a 2-degree polynomial
        func_1 = lambda x, a, b, c, s: (a * (x - s) ** 2/2 + b * (x - s) + c)
        a = np.repeat(0.002*random_gen.randn(size)+0.05, nb_time_p).reshape(size, nb_time_p).T
        b = np.repeat(a[0,:] * random_gen.randn(size) - 11 * a[0,:], nb_time_p).reshape(size, nb_time_p).T
        c = np.repeat(0.5 * random_gen.randn(size) + 2, nb_time_p).reshape(size, nb_time_p).T
        w = 10
        s = random_gen.uniform(-w, w, size)
        sim_clust_means_unaligned = func_1(time_rep, a, b, c, s)
        sim_clust_means_aligned = func_1(time_rep, a, b, c, np.zeros(size))
    if func in (2, 3):
        # Models with 3-degree polynomials (increasing and decreasing)
        func_2 = lambda x, a, r1, r2, c, d, s: (a * (x - s) ** 3/3
                                                - (x - s) ** 2 * a * (r1 + r2)/2
                                                + (x - s) * a * r1 * r2
                                                + c * (x - s) + d)
        r1 = np.repeat(random_gen.randn(size) + 8, nb_time_p).reshape(size, nb_time_p).T
        r2 = np.repeat(random_gen.randn(size) + 12, nb_time_p).reshape(size, nb_time_p).T
        if func==3:
            a = np.repeat(1e-5 * random_gen.randn(size) + 0.003, nb_time_p).reshape(size, nb_time_p).T
            d = np.repeat(0.5*random_gen.randn(size) + 2, nb_time_p).reshape(size, nb_time_p).T
        else:
            a = np.repeat(1e-5 * random_gen.randn(size) - 0.003, nb_time_p).reshape(size, nb_time_p).T
            d = np.repeat(0.5*random_gen.randn(size) + 3, nb_time_p).reshape(size, nb_time_p).T
        c = np.repeat(2 * a[0,:] * random_gen.randn(size) + 6 * a[0,:], nb_time_p).reshape(size, nb_time_p).T
        w = 10
        s = random_gen.uniform(-w, w, size)
        sim_clust_means_unaligned = func_2(time_rep, a, r1, r2, c, d, s)
        sim_clust_means_aligned = func_2(time_rep, a, r1, r2, c, d, np.zeros(size))
    if func==4:
        # Model with a sine-based function
        func_4 = lambda x, a, b, c, s: (a * np.sin(b* (x-s)) + c)
        w = 7
        s = random_gen.uniform(-w, w, size)
        a = np.abs(np.repeat(random_gen.randn(size)+2, nb_time_p).reshape(size, nb_time_p).T)
        b = np.repeat(random_gen.uniform(0.3, 0.5, size), nb_time_p).reshape(size, nb_time_p).T
        c = np.repeat(0.5*random_gen.randn(size) + 2, nb_time_p).reshape(size, nb_time_p).T
        sim_clust_means_unaligned = func_4(time_rep, a, b, c, s)
        sim_clust_means_aligned = func_4(time_rep, a, b, c, np.zeros(size))
    return sim_clust_means_unaligned, sim_clust_means_aligned
    
    
def perform_simulation_study(k, sim_clusters, sim_means, sim_cov, time_points,
                             nb_sim_rep, random_gen=None, nb_cl_rep=50,
                             max_warp_step = 2, sparsity = 0.8, n_init_sbm=2):
    """
    Performs a simulation study focusing on alignment, SBM and UMAP-based 
    clustering.

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
        points at which fold changes were simulateded.
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
        Number of random initialization attempts (k-means clustering
        initializations performed on the UMAP projection if
        method=='umap'). The default is 50.
    max_warp_step : int, optional
        If max_warp_step=i>0, then the set of all considered warps is the
        set of all integers between -i and i. The default is 2.
    sparsity : float, optional
        Sparsity of the network determining the cutoff when defining the
        binary adjacency matrix based on the weighted one. The default is 0.8.
    n_init_sbm : int, optional
        Number of initializations of SBM. The default is 2.

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
    cl_kmed_sim = Clustering(means=sim_means, cov=sim_cov,
                             time_points=time_points, dist='d2hat',
                             random_gen=random_gen)
    cl_kmed_sim_tw = Clustering(means=sim_means, cov=sim_cov,
                                time_points=time_points, dist='d2hat',
                                time_warp=True, max_warp_step=max_warp_step,
                                random_gen=random_gen)
    
    sim_net = NetworkInference(means=sim_means, cov=sim_cov,
                               time_points=time_points, dist='d2hat',
                               time_warp=True, max_warp_step=max_warp_step,
                               sparsity=sparsity, random_gen=random_gen)
    
    sim_results_ars = np.zeros((nb_sim_rep, 6))
    sim_results_vm = np.zeros((nb_sim_rep, 6))
    sim_results_cost = np.zeros((nb_sim_rep, 6))

    alg = 'k-means-like'
    for i in range(nb_sim_rep):
        ### d2hat K-medoid clustering of simulated data:
        ### WITHOUT time warping:
        (k_clusters,
         k_centroids,
         k_cost) = cl_kmed_sim.fc_clustering(k, nb_rep=nb_cl_rep, 
                                             disp_plot=False, algorithm=alg)
        sim_results_cost[i, 0] = cl_kmed_sim.calculate_comparable_cost(k,
                                                                       k_clusters)
        sim_results_cost[i, 0] /= len(time_points)
        sim_results_ars[i, 0] = adjusted_rand_score(sim_clusters, k_clusters)
        sim_results_vm[i, 0] = v_measure_score(sim_clusters, k_clusters)
        ### d2hat K-medoid clustering of simulated data :
        ### WITH time warping:
        (k_clusters_tw,
         k_centroids_tw,
         k_warps,
         k_cost_tw) = cl_kmed_sim_tw.fc_clustering(k, nb_rep=nb_cl_rep, 
                                                   disp_plot=False, 
                                                   algorithm=alg)
        sim_results_cost[i, 1] = cl_kmed_sim_tw.calculate_comparable_cost(k,
                                                                          k_clusters_tw)
        sim_results_ars[i, 1] = adjusted_rand_score(sim_clusters, k_clusters_tw)
        sim_results_vm[i, 1] = v_measure_score(sim_clusters, k_clusters_tw)
        ### SBM initialized from k-medoids:
        (sbm_sim,
         sbm_sim_centroids,
         sbm_sim_comp_cost) = sim_net.infer_sbm(k, k_clusters_tw,
                                                n_init=n_init_sbm,
                                                verbosity=0, pi_weight=0.8,
                                                random_gen=random_gen)
        clusters_sbm = sbm_sim.labels
        sim_results_cost[i, 2] = sbm_sim_comp_cost
        sim_results_ars[i, 2] = adjusted_rand_score(sim_clusters, clusters_sbm)
        sim_results_vm[i, 2] = v_measure_score(sim_clusters, clusters_sbm)
        ### SBM based on random initializations:
        (sbm_sim_random,
         sbm_sim_centroids_random,
         sbm_sim_comp_cost_random) = sim_net.infer_sbm(k, k_clusters_tw,
                                                       n_init=n_init_sbm,
                                                       verbosity=0, random=True,
                                                       random_gen=random_gen)
        clusters_sbm_random = sbm_sim_random.labels
        #print(clusters_sbm_random)
        sim_results_cost[i, 3] = sbm_sim_comp_cost_random
        sim_results_ars[i, 3] = adjusted_rand_score(sim_clusters, clusters_sbm_random)
        sim_results_vm[i, 3] = v_measure_score(sim_clusters, clusters_sbm_random)
        ### K-means clustering of the UMAP of the d2hat distance matrix:
        ### WITHOUT time warping:
        (k_clusters_umap,
         k_centroids_umap) = cl_kmed_sim.fc_clustering(k, nb_rep=nb_cl_rep,
                                                       disp_plot=False,
                                                       method='umap')
        sim_results_cost[i, 4] = cl_kmed_sim.calculate_comparable_cost(k,
                                                                       k_clusters_umap)
        sim_results_cost[i, 4] /= len(time_points)
        sim_results_ars[i, 4] = adjusted_rand_score(sim_clusters, k_clusters_umap)
        sim_results_vm[i, 4] = v_measure_score(sim_clusters, k_clusters_umap)
        ### K-means clustering of the UMAP of the d2hat distance matrix:
        ### WITH time warping:
        (k_clusters_umap_tw,
         k_centroids_umap_tw,
         k_warps_umap) = cl_kmed_sim_tw.fc_clustering(k, nb_rep=nb_cl_rep,
                                                      disp_plot=False,
                                                      method='umap')
        sim_results_cost[i, 5] = cl_kmed_sim_tw.calculate_comparable_cost(k,
                                                                          k_clusters_umap_tw)
        sim_results_ars[i, 5] = adjusted_rand_score(sim_clusters, k_clusters_umap_tw)
        sim_results_vm[i, 5] = v_measure_score(sim_clusters, k_clusters_umap_tw)
        
    print(f'd2hat without TW: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 0], axis=0)}, std {np.std(sim_results_ars[:, 0], axis=0)}')
    print(f'  V-measure score: mean {np.mean(sim_results_vm[:, 0], axis=0)}, std {np.std(sim_results_vm[:, 0], axis=0)}')
    print(f'  Cost (per time point): mean {np.mean(sim_results_cost[:, 0], axis=0)}, std {np.std(sim_results_cost[:, 0], axis=0)}')
    
    print(f'd2hat with TW: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 1], axis=0)}, std {np.std(sim_results_ars[:, 1], axis=0)}')
    print(f'  V-measure score: mean {np.mean(sim_results_vm[:, 1], axis=0)}, std {np.std(sim_results_vm[:, 1], axis=0)}')
    print(f'  Cost (per time point): mean {np.mean(sim_results_cost[:, 1], axis=0)}, std {np.std(sim_results_cost[:, 1], axis=0)}')
    
    print(f'kmed SBM: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 2], axis=0)}, std {np.std(sim_results_ars[:, 2], axis=0)}')
    print(f'  V-measure score: mean {np.mean(sim_results_vm[:, 2], axis=0)}, std {np.std(sim_results_vm[:, 2], axis=0)}')
    print(f'  Cost (per time point): mean {np.mean(sim_results_cost[:, 2], axis=0)}, std {np.std(sim_results_cost[:, 2], axis=0)}')
    
    print(f'random SBM: \n Adj. rand score: mean {np.mean(sim_results_ars[:, 3], axis=0)}, std {np.std(sim_results_ars[:, 3], axis=0)}')
    print(f'  V-measure score: mean {np.mean(sim_results_vm[:, 3], axis=0)}, std {np.std(sim_results_vm[:, 3], axis=0)}')
    print(f'  Cost (per time point): mean {np.mean(sim_results_cost[:, 3], axis=0)}, std {np.std(sim_results_cost[:, 3], axis=0)}')
    
    print(f'd2hat UMAP + k-means (no TW): \n Adj. rand score: mean {np.mean(sim_results_ars[:, 4], axis=0)}, std {np.std(sim_results_ars[:, 4], axis=0)}')
    print(f'  V-measure score: mean {np.mean(sim_results_vm[:, 4], axis=0)}, std {np.std(sim_results_vm[:, 4], axis=0)}')
    print(f'  Cost: mean {np.mean(sim_results_cost[:, 4], axis=0)}, std {np.std(sim_results_cost[:, 4], axis=0)}')
    
    print(f'd2hat UMAP + k-means (with TW): \n Adj. rand score: mean {np.mean(sim_results_ars[:, 5], axis=0)}, std {np.std(sim_results_ars[:, 5], axis=0)}')
    print(f'  V-measure score: mean {np.mean(sim_results_vm[:, 5], axis=0)}, std {np.std(sim_results_vm[:, 5], axis=0)}')
    print(f'  Cost: mean {np.mean(sim_results_cost[:, 5], axis=0)}, std {np.std(sim_results_cost[:, 5], axis=0)}')
    
    return sim_results_cost, sim_results_ars, sim_results_vm


rand_seed = 383
random_gen = np.random.RandomState(rand_seed)

nb_var = 300
time_points = np.array([ 0.5,  3 ,  6 ,  9 ,  12 ,  15 , 18 , 21. ])
nb_time_p = len(time_points)

######################### Time Warping & SBM Study 2 ##########################
k = 4
sim_clusters = random_gen.choice(range(k), nb_var)
random_gen.shuffle(sim_clusters)

### Simulating means of k clusters:
plot_time_points = np.linspace(0,21,43)
sim_means = np.zeros((nb_time_p, nb_var))
time_p_ind = np.array((1, 6, 12, 18, 24, 30, 36, 42))
time_points = plot_time_points[time_p_ind]
fig, axs = plt.subplots(2, k, figsize=(20, 16), sharey=False)
for i, cl in enumerate(range(k)):
    cluster_i = np.argwhere(sim_clusters==cl)
    (sim_means_plot_ua,
     sim_means_plot_a) = simulate_cluster_means(cluster_i.size,
                                                plot_time_points, cl+1,
                                                random_gen=random_gen)
    axs[0][i].plot(plot_time_points, sim_means_plot_ua)
    sim_means[:, np.squeeze(cluster_i)] = sim_means_plot_ua[time_p_ind, :]
    axs[1][i].plot(plot_time_points, sim_means_plot_a)
    
fig.suptitle('Study 2: functional representation of simulated means: unaligned (top) and aligned (bottom)',
             fontsize=16)
plt.tight_layout()
plt.show()

### Simulating variances of k clusters:
sim_cov = np.zeros((nb_time_p, nb_var, nb_var))
sim_var = (np.repeat(np.abs(random_gen.randn(nb_var)), nb_time_p)
           .reshape((nb_var, nb_time_p)).T)
sim_cov[:, np.arange(nb_var), np.arange(nb_var)] = sim_var

print('Study 2: the effects of alignment, SBM and UMAP-based clustering.')
nb_sim_rep = 10
(sim_results_cost,
 sim_results_ars,
 sim_results_vm) = perform_simulation_study(k, sim_clusters, sim_means,
                                            sim_cov, time_points, nb_sim_rep,
                                            random_gen=random_gen)



