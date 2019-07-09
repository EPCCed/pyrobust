"""Template problem for robust optimisation

Authors: N. Banglawala, EPCC, 2019
License: MIT

"""

###############################################################################

import numpy as np

###############################################################################


class RobustTemplateProblem(object):
    """Template class for robust problems. The compulsory methods defined here 
    must be included in all robust problem classes.  


    Attributes
    ----------
    These attributes must be set 

    bounds : list of lists, dtype=float
        list of lower and upper bounds of problem. First list gives lower
        bounds for each dimension, second list gives upper bounds for each
        dimension. 

    disturbance_bounds : list of lists, dtype=float                                           
        list of lower and upper bounds of disturbance neighbourhood. First 
        list gives lower bounds for each dimension, second list gives upper 
        bounds for each dimension.

    fitness_func : func
        fitness or objective function. Should take multi-dimensional ndarray
        as input and output single-valued real 


    """

    def __init__(self, bounds, disturbance_bounds, fitness_func) 

        self.lbound = np.array(bounds[0])
        self.ubound = np.array(bounds[1])
        self.dlbound = np.array(disturbance_bounds[0])
        self.dubound = np.array(disturbance_bounds[1]) 
        self.fitness = fitness_func
  

    # compulsory method
    # input: None
    # output: tuple(lower bounds, upper bounds) 
    #
    def get_bounds(self):
        return self.lbound, self.ubound


    # compulsory method
    # input: None
    # output: tuple(lower disturbance bounds, upper disturbance bounds)
    #
    def get_disturbance_bounds(self):
        return self.dlbound, self.dubound


    # compulsory method
    # returns fitness of individual x
    # input: x, ndarray, dtype=float
    # output: fitness of x, float
    #
    def calc_fitness(self, x):
        fitness_value = self.fitness(x)
        return fitness_value


    # compulsory method
    # method to create and evolve new individual from parents 
    # input: parents, ndarray of ndarray, dtype=float (default is 2 parents)
    #        e.g. parent 1 is parents[0, :], parent 2 is parents[1, :]
    # output: new x, ndarray, dtype=float 
    #
    def create_new_individual(self, parents):
        x = np.array([])
        return x


    # compulsory method
    # !!! DO NO EDIT !!!
    # wrapper for adding individual to population
    # input: population index of x, int
    # output: None
    #
    def add_individual(self, x_idx):
        pass


