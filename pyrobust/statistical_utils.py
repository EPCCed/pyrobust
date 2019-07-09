"""Statistical methods used in (evolutionary) optimisation problems.

PYTHON version based on source code from University of Exeter
J. E. Fieldsend, K. Alyahya, K. Doherty

Authors: N. Banglawala, EPCC, 2019
License: MIT

"""

import sys

import numpy as np
import scipy as sp

import pyDOE as pydoe
import sobol_seq


###############################################################################
# BOUNDS CHECKING


def _is_in_bounds(x, low, high):
    """Returns True if low < x < high."""

    return (np.all(x > low) and np.all(x < high))


def _if_not_in_bounds_assign_bound(x, low, high): 
    """If x > high, x = high, if x < low, x = low, else x."""
    
    if np.any(x < low):
        return low
    if np.any(x > high):
        return high
    else:
        return x                                          


###############################################################################
# RANDOM SAMPLE GENERATION METHODS


# MATLAB: sobolset(), matlab function                                           
def generate_sobol_sample(low, high, shape, **params): 
    """Generate n-dimensional Sobol sequence.                                   
                                                                                
    Parameters                                                                  
    ----------                                                                  
                                                                                
    low: float
        lower bound, not used

    high : float
        upper bound, not used

    shape :  tuple, expect 2d only as (n_points, dims)   
        shape of ndarray of random samples

    Returns                                                                     
    -------                                                                     
    ndarray, shape=(n_points, ndim) of Sobol samples                     
                                                                                
    """           
 
    try:                                                              
        n_points = shape[0]
        ndims = shape[1]                                                 
                               
        return sobol_seq.i4_sobol_generate(ndims, n_points)

    except: 
        print("Unexpected error:", sys.exc_info()[0])
        raise


def generate_latin_sample(low, high, shape, criterion='maximin',
                                     **params):
    """Latin Hypercube sampling 

    Parameters                                                                  
    ----------                                                                  
    low : float                                                                
        lower bound, not used                                         
                                                                                
    high : float                                                               
        upper bound, not used

    shape: tuple, (number of samples, number of dimensions)
        shape of ndarray of random samples

    criterion: {'center', 'maximin', 'centermaximin', 'correlation'},
               default: 'maximin'
        how to sample points, see pyDOE reference for more information:
        https://pythonhosted.org/pyDOE/randomized.html

    Returns
    -------
    ndarray, shape=shape, of Latin Hypercube samples

    Note
    ----
    In pyDOE, number of sample is number of samples per factor, and 
    number of dimensions is number of factors, given (ndims, no. of samples) 

    """    

    try:
        ndims = shape[1]
        num_samples = shape[0]
       
        sample = pydoe.lhs(ndims, num_samples, criterion=criterion)

        #print("\nDEBUG: LHS sample {}".format(sample))

        return sample                 
                                                               
    except:                                                                     
        print("Unexpected error:", sys.exc_info()[0])                                                  
        raise



# dictionary of sampling methods
SAMPLING_ = dict(uniform=np.random.uniform,                             
                     sobol=generate_sobol_sample,                     
                     latin=generate_latin_sample) 


def generate_random_sample(low, high, shape, sampling='uniform', **params):
        """Generates ndarray of floats using rnd_type sampling method. 

        Parameters
        ----------
        low: float
            lower bound  
 
        high: float
            lower bound

        sampling: str, see dict SAMPLING_, default: 'uniform'
	    sampling method to use

        shape: tuple
            shape of sample array (num of samples and dimension of each sample)
            
        Returns
        -------
        ndarray, shape=shape, of random floats

        """ 

        try:
            return SAMPLING_[sampling](low, high, shape)

        except KeyError:
            print("Sampling method must be one of: {0}, got: {1}".format(
                  SAMPLING_.keys(), sampling))
        except:
            print("Unexpected error: ", sys.exc_info()[0])                           
            raise 



