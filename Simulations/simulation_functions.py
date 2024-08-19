"""
Functions generating data for simulation studies.

@author: Polina Arsenteva
"""
import numpy as np

def simulate_cluster_means_1(size, time_points, func, random_gen=None):
    """
    Simulates means for a given cluster for the first series of simulation
    studies.

    Parameters
    ----------
    size : int
        Number of members in the cluster.
    time_points : array-like
        1D array-like containing data with `float` type, representing time
        points for the simulated dataset.
    func : int (1, 2, 3 or 4)
        Indicates a simulation model for cluster means, different models
        generate different clusters. Each model is based on a polynomial with
        random coefficients.
    random_gen : RandomState instance or None, optional
        Random number generator, used to reproduce results. If None (default),
        the generator is the RandomState instance used by `np.random`.
        If RandomState instance, random_gen is the actual random
        number generator.

    Returns
    -------
    sim_clust_means : ndarray
        2D array of shape (len(time_points), size) containing data
        with `float` type, representing fold changes' means for each simulated
        entity and each time point.

    """
    nb_time_p = len(time_points)
    time_rep = np.repeat(time_points, size).reshape(nb_time_p, size)
    if random_gen is None:
        random_gen = np.random
    if func == 1:
        # Model with a 2-degree polynomial
        def func_1(x, a, b, c): return (a*x**2/2 + b * x + c)
        a = np.repeat(0.005*random_gen.randn(size)+0.05,
                      nb_time_p).reshape(size, nb_time_p).T
        b = np.repeat(2 * a[0, :] * random_gen.randn(size) -
                      10 * a[0, :], nb_time_p).reshape(size, nb_time_p).T
        c = np.repeat(random_gen.randn(size) + 2,
                      nb_time_p).reshape(size, nb_time_p).T
        sim_clust_means = func_1(time_rep, a, b, c)
    if func in (2, 3):
        # Models with 3-degree polynomials (increasing and decreasing)
        def func_2(x, a, r1, r2, c, d): return (a*x**3/3 - x**2 * a * (r1 + r2)/2
                                                + x * a * r1 * r2 + c * x + d)
        r1 = np.repeat(random_gen.randn(size) + 5,
                       nb_time_p).reshape(size, nb_time_p).T
        r2 = np.repeat(random_gen.randn(size) + 15,
                       nb_time_p).reshape(size, nb_time_p).T
        if func == 3:
            a = np.repeat(0.001 * random_gen.randn(size) + 0.01,
                          nb_time_p).reshape(size, nb_time_p).T
            d = np.repeat(random_gen.randn(size) + 3,
                          nb_time_p).reshape(size, nb_time_p).T
        else:
            a = np.repeat(0.001 * random_gen.randn(size) - 0.01,
                          nb_time_p).reshape(size, nb_time_p).T
            d = np.repeat(random_gen.randn(size) + 2,
                          nb_time_p).reshape(size, nb_time_p).T
        c = np.repeat(2 * a[0, :] * random_gen.randn(size) +
                      6 * a[0, :], nb_time_p).reshape(size, nb_time_p).T
        sim_clust_means = func_2(time_rep, a, r1, r2, c, d)
    if func == 4:
        # Model with a 4-degree polynomial
        def func_4(x, a, r1, r2, r3, b, c): return (a * (x**4/4 + x**3 * (-r1-r2-r3)/3
                                                         + x**2 *
                                                         (r1 * r2 + r3 *
                                                          (r1 + r2))/2
                                                         - x * r1 * r2 * r3) + b * x + c)
        r1 = np.repeat(0.2*random_gen.randn(size)+2,
                       nb_time_p).reshape(size, nb_time_p).T
        r2 = np.repeat(0.5*random_gen.randn(size)+10,
                       nb_time_p).reshape(size, nb_time_p).T
        r3 = np.repeat(0.2*random_gen.randn(size)+18,
                       nb_time_p).reshape(size, nb_time_p).T
        a = np.repeat(5e-5*random_gen.randn(size)+5e-3,
                      nb_time_p).reshape(size, nb_time_p).T
        #b = np.repeat(0.05*random_gen.randn(size), nb_time_p).reshape(size, nb_time_p).T
        b = np.repeat(random_gen.uniform(-0.05, 0.05, size),
                      nb_time_p).reshape(size, nb_time_p).T
        c = np.repeat(0.5*random_gen.randn(size)+2,
                      nb_time_p).reshape(size, nb_time_p).T
        sim_clust_means = func_4(time_rep, a, r1, r2, r3, b, c)
    return sim_clust_means

def simulate_cluster_means_2(size, time_points, func, random_gen=None):
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
    
