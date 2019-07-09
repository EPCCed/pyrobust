"""Default problem for robust optimisation, based on:

Title:   On the Exploitation of Search History and Accumulative Sampling in
         Robust Optimisation
Authors: K. Doherty, K. Alyahya, J. E. Fieldsend, O. E. Akman
Journal: GECCO '17 Companion, Berlin, Germany
DOI:     http://dx.doi.org/10.1145/3067695.3076060
  
PYTHON version based on source code from University of Exeter
J. E. Fieldsend, K. Alyahya, K. Doherty

Authors: N. Banglawala, EPCC, 2019
License: MIT

"""

###############################################################################

import numpy as np

from pyrobust import statistical_utils as su
from pyrobust import evolving_methods as em
from pyrobust import resampling_methods as rm
from pyrobust import weighting_methods as wm


###############################################################################


class RobustDefaultProblem(object):
    """
       Notes
       -----
       The default problem supplies evolution methods for creating new 
       individuals from 2 parents: simulated binary crossover and (Gaussian) 
       mutation (see evolving_methods module). Only the problem bounds, the 
       fitness (objective function) need to be supplied. Other parameters, such
       as those controlling evolution (crossover / mutation) and the elite
       population size have defaults but can also be specified. 

    """

    def __init__(self, bounds, disturbance_bounds, fitness_func, 
                 elite_pop_size=20, crossover_rate=0.8, mutation_rate=0.5, 
                 mutation_width=0, max_tries=100, SBX_N=20):

        self.lbound = np.array(bounds[0])
        self.ubound = np.array(bounds[1])
        self.dlbound = np.array(disturbance_bounds[0])
        self.dubound = np.array(disturbance_bounds[1]) 
        self.fitness = fitness_func
 
        self.elite_pop_size = 20
        # number of parents to select
        self.num_select = 2 

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        if mutation_width:
            self.mutation_width = mutation_width
        else:
            self.mutation_width = \
                        0.1 * abs(np.array(bounds[1]) - np.array(bounds[0]))

        self.max_tries = max_tries
        self.SBX_N = SBX_N 
       
        return 


    def get_bounds(self):
        return self.lbound, self.ubound


    def get_disturbance_bounds(self):
        return self.dlbound, self.dubound


    def calc_fitness(self, x):
        fitness_value = self.fitness(x)
        return fitness_value


    # creates new individual 
    def create_new_individual(self, parents):
        # crossover method
        crossover_func = em.simulated_binary_crossover

        # mutation method
        mutation_func = em.gaussian_mutation

        # crossover parents to produce new individual   
        x_new = em.crossover(parents, self.crossover_rate, crossover_func,
                             lbound=self.lbound, ubound=self.ubound,
                             max_tries=self.max_tries, SBX_N=self.SBX_N)

        # randomly mutate x_new
        x_mut = em.mutate(x_new, self.mutation_rate, mutation_func,
                          mutation_width=self.mutation_width)

        # while mutated individual not within bounds, keep mutating
        while not su._is_in_bounds(x_mut, self.lbound, self.ubound):
            x_mut = em.mutate(x_new, self.mutation_rate, mutation_func,
                              mutation_width=self.mutation_width)

        return x_mut


    # !!! DO NO EDIT !!!
    # wrapper for adding individual to population
    def add_individual(self, x_idx):
        pass

