"""Weighting Methods.

PYTHON version based on source code from University of Exeter
J. E. Fieldsend, K. Alyahya, K. Doherty


Authors: N. Banglawala, EPCC, 2019
License: MIT

"""

import sys

import numpy as np
import scipy as sp

from pyrobust import statistical_utils as su


###############################################################################


def uniform_weights(x_epts, cpts=None, lbound=0, ubound=0, num_cpts=0, **params):
    """Calculates uniform weights.

    Parameters
    ----------
    x_epts : ndarray, dtype=float
        evaluation or archive points

    cpts : ndarray, float, default: None
        candidate (reference) points

    lbound : float, default: 0
        lower bound

    ubound : float, default: 0
        upper boun

    num_cpts : int, default: 0
        number of candidate points to generate


    Returns
    -------
    weights : ndarray, dtype=float

    """

    assert x_epts.ndim > 1, print("Need > 1 point for uniform weights")

    num_x_epts = x_epts.shape[0]

    return  np.ones(num_x_epts)/num_x_epts


def wasserstein_weights(x_epts, cpts=None, lbound=0, ubound=0, num_cpts=0,
                        **params):
    """Calculates Wasserstein weights.

    Parameters
    ----------
    x_epts : ndarray, dtype=float
        evaluation or archive points

    cpts : ndarray, float, default: None
        candidate (or reference) points

    lbound : float, default: 0
        lower bound

    ubound : float, default: 0
        upper boun

    num_cpts : int, default: 0
        number of candidate points to generate

    Returns
    -------
    weights : ndarray, dtype=float

    """

    assert x_epts.ndim > 1, print("Need > 1 point for wasserstein weights")

    try:
        # generate candidate points if none given
        if cpts is None:
            # generate sample, if parameters set
            if num_cpts and (ubound - lbound):
                dims = x_epts.shape[1]
                cpts_shape = (num_cpts, dims)
                # low /high bounds not used in su.generate_latin_sample
                cpts = su.generate_latin_sample(0, 1, cpts_shape)
                cpts = cpts * (ubound - lbound) + lbound
            else:
                print("Could not generate candidate points")
                return

        num_x_epts = x_epts.shape[0]
        num_cpts = cpts.shape[0]
        closest = np.zeros((num_cpts, num_x_epts))

        # distances between each x point and candidate point
        distances = sp.spatial.distance.cdist(cpts, x_epts)

        # index of x_ept that is closest to a given cpt
        min_idx = np.argmin(distances,1)

        # count how many times an x_ept is the nearest neighbour to a cpt
        closest[:, min_idx] = 1
        num_nn = np.sum(closest,0)

        num_cpts = cpts.shape[0]
        return num_nn/num_cpts

    except:
        print("Unexpected error: ", sys.exc_info()[0])
        raise


def voronoi_monte_carlo_weights(x_epts, cpts=None, lbound=0, ubound=0,
                                num_cpts=0, **params):
    """Calculates Voronoi weights using Monte Carlo sampling."""

    pass
