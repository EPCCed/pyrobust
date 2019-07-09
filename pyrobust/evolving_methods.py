"""Methods for creating and evolving populations.                                                  
                                                                                
PYTHON version based on source code from University of Exeter                   
J. E. Fieldsend, K. Alyahya, K. Doherty                                           
                                                                                
see pop_project/src/GECCO_2017/EAS_GA_UpdateHistory                             
                                                                                
Authors: N. Banglawala, EPCC, 2019                                              
License: MIT                                            
                                                                                
"""                                                                             
                                                                                
import sys                                                                      
                                                                                
import numpy as np

from pyrobust import statistical_utils as su                                                          
                                                                                                                            
############################################################################### 

#======= MUTATION METHODS

def mutate(x, mutation_rate, mutation_func, **params):
    """Wrapper for mutation methods.                                        
                                                                            
    Parameters                                                              
    ----------                                                              
    x : ndarray, dtype=float                                                
        individual to mutate

    mutation_rate : float
        mutation rate

    mutation_func : func
        mutation method                                              
                                                                            
    """    

    assert x.ndim == 1, \
           print("Mutate only one individual at a time, got", x)                                                                 
                    
    dims = x.shape[0]
    
    # randomly choose which elements of x to mutate
    to_mutate = np.random.ranf(dims)                                   
                                                                             
    # if no element of x chosen to be mutated, choose one at random             
    to_mutate_idx = (to_mutate < mutation_rate) 
      
    # if not element is chosen to mutate, pick one                                                                          
    if not np.any(to_mutate_idx):                                           
        to_mutate_idx[np.random.randint(0, dims, 1)] = 1                  
                                                                                
    return mutation_func(x, to_mutate_idx, **params)    


def gaussian_mutation(x, to_mutate_idx, mutation_width):                            
    """Mutates individual x using gaussian distribution with standard       
    deviation mutation_width.                                               
                                                                            
    Parameters                                                              
    ----------                                                              
    x : ndarray, dtype=float                                                
        individual to mutate                                                
                                                                            
    to_mutate_idx : int or ndarray, dtype=int                               
        indices of elements of x to mutate                                  
                                                      
    mutation_width : float or ndarray, dtype=float
        width of 1d Gaussian to sample from
                      
    Returns                                                                 
    -------                                                                 
    x mutated.                                                              
                                                                            
    """                                                                     
                                                                            
    try:                                                                    
                                                                            
        mutations = \
        np.random.normal(0, mutation_width, x.size) * to_mutate_idx    
                                                                            
        x = x + mutations                                                   
        return x
                                                            
    except:                                                                 
        print("Unexpected error:", sys.exc_info()[0])                       
        raise                                                               
                                                          


#======= CROSSOVER / RECOMBINATION METHODS
                                        
def crossover(parents, crossover_rate, crossover_func, **params): 
    """Wrapper for crossover methods.                                       
                                                                            
    Parameters                                                              
    ----------                                                              
    parents : ndarray, dtype=float                                                   
        parents (at least 1)                                            

    crossover_rate : float
        crossover rate

    crossover_func : func
        crossover method      
                                                                            
    Returns                                                                 
    -------                                                                 
    x_new : ndarray, dtype=float                                            
        new recombined individual                                           
                                                                            
    """                                                                     
                                                                            
    if np.random.ranf() < crossover_rate:                              
        return crossover_func(parents, **params)                 
    else:                                               
        return parents[0]


def simulated_binary_crossover(parents, lbound, ubound, max_tries=100, 
                               SBX_N=20):
    """Simulated binary crossover, crosses two parents to produce new    
    individual.                    
                                                                                
    Parameters                                                              
    ----------                                                              
    parents : ndarray, ndarray, dtype=float                                                   
        2 parents - will take first two parents if more than 2 given           

    ubound, lbound : ndarray, dtype=float
        allowed upper and lower bounds of x

    max_tries : int, default: 100
        number of tries to generate valid recombined values

    SBX_N : int, default: 20
        constant
                                                                                
    Returns                                                                 
    -------                                                                 
    x_new : ndarray, dtype=float          
       new recombined individual                                           
                                                                           
    """                                                                     

    dims = len(lbound)

    # take first 2 parents if more than 2 given    
    x1 = parents[0, :]
    x2 = parents[1, :]

    inv_SBX_N = 1/(SBX_N  + 1)                                              
    bounds_diff = np.abs(ubound - lbound) 

    rnd0 = np.random.ranf()                                                 
                                                                           
    # initialise x_new as parent clone 
    x_new = x1                                                              
                                                                                
    for i in range(0, dims):                                           
        tries = 0                                                           
                                                                            
        if np.random.ranf() < 0.5:                                          
            valid = 0                                                       
            while not valid and (tries <= max_tries):                       
                tries = tries + 1                                           
                rnd1 = np.random.ranf()                                     
                if rnd1 < 0.5:                                              
                    beta = (2 * rnd1)**(inv_SBX_N)                          
                else:                                                       
                    beta = (0.5/(1-rnd1))**(inv_SBX_N)                      

                
                                                                            
                mean_p = 0.5 * (x1[i] + x2[i])                              
                c1 = mean_p - 0.5 * beta * abs(x1[i] - x2[i])               
                c2 = mean_p + 0.5 * beta * abs(x1[i] - x2[i])

                if np.all(c1 <= lbound) and np.all(c1 >= ubound):
                    if np.all(c2 <= lbound) and np.all(c2 >= ubound):
                        valid = 1                                                     
                                 
            c1 = su._if_not_in_bounds_assign_bound(c1, lbound[i], 
                                                      ubound[i])           
            c2 = su._if_not_in_bounds_assign_bound(c2, lbound[i],        
                                                      ubound[i])           
                                                                               
            if rnd0 < 0.5:                                                  
                x_new[i] = c1                                               
            else:                                                           
                x_new[i] = c2                                               
                                                                                
    return x_new 
