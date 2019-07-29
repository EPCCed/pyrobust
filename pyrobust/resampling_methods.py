"""Resampling methods.

PYTHON version based on source code from University of Exeter
J. E. Fieldsend, K. Alyahya, K. Doherty

see pop_project/src/GECCO_2017/EAS_GA_UpdateHistory

Authors: N. Banglawala, EPCC, 2019
License: MIT

"""

import sys

import numpy as np
import scipy as sp
import scipy.spatial.distance as spdist

from pyrobust import statistical_utils as su
from pyrobust import weighting_methods as wm


###############################################################################


def uniform_resampling(x0, xpts, lbound, ubound, num_cpts,
                       weighting=wm.uniform_weights, **params):
    """Uniform resampling.

    Parameters
    ----------
    x0 : ndarray, dtype=float
        source point

    xpts : ndarray, dtype=float
        evaluation points in neighbourhood of x0

    lbound : float
        lower bound

    ubound : float
        upper bound

    num_cpts : int
        number of candidate or reference points

    weighting_method : func, default: wm.uniform_weights
        weighting method

    Returns
    -------
    x_new : ndarray, dtype=float
        new resampled point

    weights : ndarray, dtype=float
        associated weights

    """

    try:
        dims = xpts.shape[1]

        num_cpts = 1

        cpts_shape = (num_cpts, dims)
        x_new = \
            x0 + np.random.uniform(0, 1, cpts_shape) * (ubound - lbound) \
               + lbound

        x_epts = np.vstack([xpts, x_new])  

        weights = weighting(x_epts, cpts=xpts, lbound=lbound, ubound=ubound, 
                            num_cpts=num_cpts)

        return x_new, x_epts, weights

    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise



def sobol_resampling(x0, xpts, lbound, ubound, num_cpts,
                     weighting_method=wm.uniform_weights, **params):
    """Sobol resampling.

    Parameters
    ----------
    x0 : ndarray, dtype=float
        centre point

    xpts : ndarray, dtype=float
        evaluation points in neighbourhood of x0

    lbound : float
        lower bound

    ubound : float
        upper bound

    num_cpts : int
        number of candidate or reference points

    weighting_method : func, default: wm.uniform_weights
        weighting method

    Returns
    -------
    x_new : ndarray, dtype=float
        new resampled point

    weights : ndarray, dtype=float
        associated weights

    """

    try:
        dims = xpts.shape[1]

        cpts_shape = (num_cpts, dims)

        cpts = su.generate_sobol_sample(lbound, ubound, cpts_shape)

        # randomly permute indices for each sample
        idx_perm = np.random.permutation(np.arange(num_cpts))
        x_new = x + cpts[np.random.randi(0,num_cpts,idx_perm), :]

        weights = weighting_methods(xpts)

        return x_new, weights, cpts

    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def wasserstein_resampling(x0, xN, lbound, ubound, num_cpts,
                           weighting=wm.wasserstein_weights):
    """Wasserstein resampling.

    Parameters
    ----------
    x0 : ndarray, dtype=float
        centre point

    xN : ndarray, dtype=float
        evaluation points in neighbourhood of x0

    lbound : float
        lower bound

    ubound : float
        upper bound

    num_cpts : ndarray, dtype=float
        number of candidate or reference points

    weighting : func, default: wm.wasserstein_weights
        method for generating weights

    Returns
    -------
    weights : ndarray, dtype=float
        Wasserstein weights

    x_new : ndarray, dtype=float
        new evaluated point

    cpts : ndarray, dtype=float
        candidate or reference points

    """

    try:
        dims = xN.shape[1]
        cpts_shape = (num_cpts, dims)

        # generation samples (note bounds in latin sample do nothing)
        sample = su.generate_latin_sample(0, 1, cpts_shape)

        # candidate points
        cpts = \
            np.tile(x0, (num_cpts, 1)) + sample * (ubound - lbound) + lbound
        
        # complete set of evaluation points in x neighbourhood
        x_epts = np.vstack([x0, xN])

        # total distance between each cpt and closest member of x_ept,
        # x_epts's neighbours and other candidate points
        tot_min_dists = np.zeros(num_cpts, dtype=float)

        # distances between all cpts and x_epts
        all_dists = spdist.cdist(cpts, x_epts)

        # pairwise distances between cpts
        cpts_dists = spdist.cdist(cpts, cpts)

        # dist of x_ept to its nearest neighbour candidate point
        all_min_dists = np.min(all_dists, 1)


        for i in range(num_cpts):

            cpt_i = cpts[i, :]

            # index without ith cpt
            no_i_idx = np.delete(np.arange(num_cpts),i)

            cpts_no_i = np.array([cpts_dists[no_i_idx, i]])
            cpts_no_i.shape = (cpts_no_i.size, 1)

            # all evaluation points except ith candidate point
            all_epts_no_i = \
                np.hstack((all_dists[no_i_idx, :], cpts_no_i))
            min_dists = np.min(all_epts_no_i, 1)

            # wasserstein distance
            tot_min_dists[i] = np.sum(min_dists)

        # find best cpts, one that is closest to the largest number of x_epts
        # minimise Wasserstein distance
        best_idx = np.argmin(tot_min_dists)

        x_new = cpts[best_idx, :]

        not_best_idx = np.delete(np.arange(num_cpts), best_idx)

        # new set of evaluation points for weights
        x_epts_new = np.vstack([x_epts, x_new])

        weights = weighting(x_epts_new, cpts=cpts[not_best_idx, :],
                            lbound=lbound, ubound=ubound)

        return x_new, weights, cpts

    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def voronoi_monte_carlo_resampling(x0, xpts, lbound, ubound, num_cpts,
                                   weighting=wm.voronoi_monte_carlo_weights):
    """Voronoi resampling using the Monte Carlo method."""
    pass
