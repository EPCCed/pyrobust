"""Class for creating and manipulating populations of design variables

PYTHON version based on source code from University of Exeter
J. E. Fieldsend, K. Alyahya, K. Doherty

Authors: N. Banglawala, EPCC, 2019
License: MIT
 
"""

import sys

import numpy as np

from pyrobust import statistical_utils as su

###############################################################################

class Population(object):
    """

    Parameters
    ----------

    dims : int
        number of dimensions

    max_pop : int
        maximum number of individuals, used to set size of population arrays

    problem : class instance
        class instance for problem parameters and methods, containing at least:
            - get_bounds : getter for upper and lower bounds of problem. 
                           Note: deduce dimensions of problem from bounds
            - get_disturbance_bounds : getter for upper and lower bounds of
                                       disturbance neighbourhood     
            - fitness : objective function, takes individual and outputs its
                        fitness value
            - create_new_individual : create new individual method
            - calc_fitness : wrapper method
            - add_individual : wrapper method
 
    Attributes
    ----------
    n_pop_ : int
        current size of population i.e. last occupied index in x_

    x_ : ndarray, dtype=float, shape(max_pop, dims)
        array of design variables (individuals)

    obj_vals_ : ndarray, dtype=float, shape(max_pop, max_pop)
        of objective values for each x and all neighbours used to calculate its
        average objective i.e. obj_vals_[i][j] is the jth objective value
        contributing to obj_avg_[i] for individual x_[i].

    n_obj_ : ndarray, dtype=int, shape(max_pop,)
        number of values in obj_vals_ for each x.

    obj_avg_ : ndarray, dtype=float, shape(1, max_pop)
        obj_avg_[i] is the average objective value for individual x_[i].

    x_nbrs_ : ndarray, dtype=float, shape(max_pop, max_pop, dims)
        array of neighbours (disturbed values) of x (each row is a neighbour).

    n_nbrs_ : ndarray, dtype=int, shape(max_pop,)
        number of neighbours for each individual in population

    sort_by : self.obj_avg_
        field by which to sort entire population before any selection. Fixed
        to be obj_avg_ but in place for future flexibility

    sort_descend : bool, default False
        sort population by descending values 

    __is_sorted : bool
        True if population is in a sorted state, False otherwise

    __sorted_idx : ndarray, dtype=int, shape(max_pop,)
        sorted_idx_[i] gives the position of the individual i in a population
        sorted by field sorted_by_field in sorted_order order.


    Notes
    -----
    To find the next empty cell in a given population array:

    Next empty cell in x_,  x_[n_pop_, :]
    Next empty cell in obj_vals_ for given x_[i], obj_vals_[i, n_obj_[i]]
    Next empty cell in obj_avg_, obj_avg_[n_pop_]
    Next empty cell in x_nbrs_ for x_[i], x_nbrs_[i, n_nbrs_[i], :]
    Next empty cell in sorted_idx, sorted_idx[n_pop_]

    """


    def __init__(self, max_pop, problem, sort_descend=False, *args, **params):

        self.max_pop = max_pop
        self.problem = problem

        # bounds need to be numpy arrays, even if scalars
        self.lbound = problem.get_bounds()[0]
        self.ubound = problem.get_bounds()[1]
        self.dlbound = problem.get_disturbance_bounds()[0]
        self.dubound = problem.get_disturbance_bounds()[1]

        self.dims = len(self.lbound)
        self.in_args = args
        self.in_params = params
 
        self.sort_descend = sort_descend

        # set attributes for evolve from problem, else use defaults
        attr = ['num_parents', 'elite_pop_size', 'select_as']
        attr_defaults = [2, 10, 'random']
        for i, a in enumerate(attr):
            if hasattr(self.problem, a):
                setattr(self, a, getattr(self.problem, a))
            else:
                setattr(self, a, attr_defaults[i]) 

        # prepare and initialise population arrays etc.
        self.__prepare_population()

        return


    def __prepare_population(self):
        """Initialises population arrays and other popluation attributes."""

        if self.dims < 1 :
            raise ValueError("Number of dimensions must be positive, got",
                             self.dims)

        if self.max_pop < 1 :
            raise ValueError("Max pop size must be positive, got",
                              self.max_pop)

        self.n_pop_ = 0

        self.x_ = np.zeros((self.max_pop, self.dims), dtype=float)

        self.obj_vals_ = \
            np.zeros((self.max_pop, self.max_pop), dtype=float)
        self.n_obj_ = np.zeros((self.max_pop,), dtype=int)  
        self.obj_avg_ = np.zeros((self.max_pop,), dtype=float)
        self.x_nbrs_ = \
            np.zeros((self.max_pop, self.max_pop, self.dims), dtype=float)
        self.n_nbrs_ = np.zeros((self.max_pop,), dtype=int)

        # for sorting population
        self.sort_by = self.obj_avg_
        self.__is_sorted = False
        self.__sorted_idx = None 

        return


    def reset(self):
        """Resets population to empty state."""

        self.__prepare_population()


    def get_pop_size(self):
        return self.n_pop_      


    def get_pop(self):
        return self.x_[:self.n_pop_, :] 


    def get_individual(self, idx):
        return self.x_[idx, :]


    def get_fitness_value(self, idx, jdx):
        # if no specific jdx given, return all fitness values
        if jdx is None:
            return self.obj_vals_[idx, :self.n_obj_[idx]]
        elif jdx < 0:  
            return self.obj_vals_[idx, jdx + self.n_obj_[idx]]
        else:
            return self.obj_vals_[idx, jdx]


    def get_population_fitness_values(self):
        return self.obj_vals_[:,0]


    def get_fittest(self, num, get_idx=False):
        if not self.__is_sorted:
            self.__sort_population()
 
        best_idx = self.__sorted_idx[0:num]
        best_x = self.get_individual(best_idx)
        best_fit = self.get_fitness_value(best_idx, 0)
        best_fit_est = self.get_fitness_estimate(best_idx) 
 
        if get_idx:
            return best_x, best_fit, best_fit_est, best_idx 
       
        return best_x, best_fit, best_fit_est


    def get_least_eval(self, num):
        if not self.__is_sorted:                                                
            self.__sort_population()

        return self.n_obj_[self.__sorted_idx[0:num]]


    def get_fitness_estimate(self, idx):
        return self.obj_avg_[idx]


    def set_fitness_estimate(self, idx, val):
        self.obj_avg_[idx] = val


    def get_num_neighbours(self, idx):
        return self.n_nbrs_[idx]


    def get_num_fitness_values(self, idx):
        return self.n_obj_[idx] 


    def get_num_evaluations(self, idx):
        return len(self.obj_vals_[idx])


    def get_neighbour(self, idx, jdx):
        # if jdx is None, return all neighbours
        if jdx is None:
            return self.x_nbrs_[idx, :self.n_nbrs_[idx], :]
        elif jdx < 0:
            return self.x_nbrs_[idx, :self.n_nbrs_[idx], :] 
        else:
            return self.x_nbrs_[idx, jdx, :]


    def update_pop_size(self):
        self.n_pop_ = self.n_pop_ + 1
        return


    def add_fitness_value(self, idx, val):
            self.obj_vals_[idx, self.n_obj_[idx]] = val
            self.n_obj_[idx] = self.n_obj_[idx] + 1
            return 


    def add_neighbour(self, x_idx, nbr, nbr_fit):
        """Adds a neighbour to individual's neighbourhood. 
           Returns total number of neighbours."""

        try:  
            # add neighbour
            self.x_nbrs_[x_idx, self.n_nbrs_[x_idx]] = nbr

            # update neighbour count
            self.n_nbrs_[x_idx] = self.n_nbrs_[x_idx] + 1

            # add neighbour fitness
            self.obj_vals_[x_idx, self.n_obj_[x_idx]] = nbr_fit

            # update obj_vals count
            self.n_obj_[x_idx] = self.n_obj_[x_idx] + 1
          
            return self.n_obj_[x_idx]

        except:
            print("Unexpected error : ", sys.exc_info()[0])
            raise
 
  
    def _create_initial_population(self, init_sample):
        """Creates initial population from init_sample.

        Parameters
        ----------
        init_sample : ndarray, dtype=float
            individuals to add to population. Note, these are only x (design)
            values, fitness values etc. need to be calculated
 
        """

        try:
            # make singleton into array of singleton

            if len(init_sample) == self.dims:
                init_sample = [init_sample]

            assert len(init_sample) <= self.max_pop, \
                print("Initial sample size must be <= max pop, got",
                       len(init_sample))
 
            for x in init_sample:
                self.add_individual(x)
            return

        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise



    def add_individual(self, x, x_fit=None, x_fit_est=None, *args, **params):
        """Wrapper for add individual method."""

        x_idx, x_fit, x_fit_est = \
            self._add_individual(x, x_fit=x_fit, x_fit_est=x_fit_est, *args,
                                 **params)
 
        return x_idx, x_fit, x_fit_est
     

    def _add_individual(self, x, x_fit=None, x_fit_est=None, *args, **params):
        """Adds individual to population, updating fitness and fitness 
        estimate.

        Parameters
        ----------
        x : ndarray, dtype=float
            individual to add

        x_fit : float, default: None
            fitness value

        x_fit_est : float, default: None               
            fitness estimate 
        """ 
 
        # get population size (index of new individual)
        pop_size = self.n_pop_
    
        # number of fitness values in individual's history
        # should be zero for new individual
        num_fit = self.n_obj_[pop_size]

        # add x to population, but do not update population size yet
        self.x_[pop_size, :] = x 

        # add fitness value
        if not x_fit:
            x_fit = self.problem.calc_fitness(x) 
           
        self.add_fitness_value(pop_size, x_fit) 
        #self.obj_vals_[pop_size, jdx] = x_fit

        if not x_fit_est:
            x_fit_est = x_fit           

        self.obj_avg_[pop_size] = x_fit_est
 
        # index of new individual
        x_idx = pop_size

        # !!! IMPORTANT !!! update population size
        self.update_pop_size()

        self.__is_sorted = False
 
        return x_idx, x_fit, x_fit_est


    def evolve_and_add_to_population(self):
        """Evolves new individual and adds to valid population if within 
        bounds."""

        x_new = self.evolve()

        if x_new is not None:
            try:
                x_idx, x_fit, x_fit_est = self.add_individual(x_new)
    
                return

            except:                                                              
                print("Unexpected error:", sys.exc_info()[0])                       
                raise 
        return


    def evolve(self):
        """Evolves new individual and add to population."""
    
        # select parents
        parents_idx = self.select_from_fittest(self.num_parents, 
                                               self.elite_pop_size,
                                               select_as=self.select_as)

        # get parents
        parents = self.x_[parents_idx]

        # create new individual from parents
        x_new = self.problem.create_new_individual(parents)

        # if x_new is not within bounds
        if not su._is_in_bounds(x_new, self.lbound, self.ubound):
            x_new = None

        return x_new


    def __select_individuals(self, num_select, elite_pop_size,                   
                             select_by='fittest', select_as='random'): 
        """Select individuals from elite (fittest) sub-population               
                                                                                
        Parameters                                                              
        ----------                                                              
        num_select : int                                                        
            number of individuals to select                                     
                                                                                
        elite_pop_size : int                                                    
            size of elite population from which to choose individuals. This     
            is the sub-population of fittest individuals. See notes.             
                                                
        select_by : {'fitness', 'least_eval'}
            either choose by fitness ('fitness') or by least number of
            evaluations ('least_eval') out of elite population.
                                
        select_as : {'random', 'top'}                                           
            choose individuals randomly, or as top num_select           
            individuals given sub-population select pool.                     
                                                         
        Returns                                                                 
        -------                                                                 
        list of population indices of selected parents                         
                                                                                
        Notes                                                                   
        -----                                                                   
        Upper bound for parent index range is set as follows:                   
        - if choosing parents from elites (elite_pop_size > 0), the upper bound 
          is elite_pop_size if the population size is >= elite_pop_size, else   
          it is the population size.                                            
                                                                                
        """                                                                     
                                                                                
        select_pool = self.n_pop_                                               
                                         
        # sort population if not in sorted stated                               
        if not self.__is_sorted:                                                
            self.__sort_population()                                

        if elite_pop_size:                                                      
            select_pool = min(elite_pop_size, self.n_pop_)                      
                                                                                
        assert select_pool >= num_select, \
            print("num_select {} > select_pool {}".format(num_select, 
                                                          select_pool)) 
                                                                           
        selected = None 
        try:                                                                    
            if select_by == 'fitness':                                          
                if select_as == 'random':
                    selected = set()                                       
                    while (len(selected) < num_select):                           
                        selected.add(np.random.randint(0, select_pool, 1)[0])
                    selected = list(selected)     
                if select_as == 'top':       
                    selected = self.__sorted_idx[:num_select] 
                                                                                
            if select_by == 'least_eval':                                      
                                                                                
                # sort subpopulation by number of evaluations (ascending)       
                least_eval_idx = \
                    np.argsort(self.n_obj_[self.__sorted_idx[:select_pool]])  
                
                # from fittest individuals, select those with fewest evaluations   
                selected= self.__sorted_idx[least_eval_idx[:num_select]] 
                                                                                
            return list(selected)  

        except:                                                                 
            print("Unexpected error:", sys.exc_info()[0])                       
            raise



    def select_from_fittest(self, num_select, elite_pop_size, 
                            select_as='random'):
        """Select individuals from elite (fittest) sub-population
 
        Parameters
        ----------
        num_select : int
            number of individuals to select

        elite_pop_size : int
            size of elite population from which to choose individuals. This 
            is the subpopulation of fittest individuals.

        select_as : {'random', 'top'}
            choose individuals randomly, or as top num_select fittest 
            individuals from elite subpopulation.  

        Returns
        -------
        list of population indices of selected parents
         
        """

        return self.__select_individuals(num_select, elite_pop_size, 
                                         select_by='fitness', 
                                         select_as=select_as)

 
    def select_least_evaluated(self, num_select, elite_pop_size):
        """Select individuals with the least number of evaluations out of 
        fittest subi-population (elite subpopulation).
        
        Parameters                                 
        ----------                                                              
        num_select : int                                                        
            number of individuals to select                                     
                                                                                
        elite_pop_size : int                                                    
            size of elite population from which to choose individuals. This  
            is the sub-population of fittest individuals. 
       
        Returns                                                                 
        -------                                                                 
        list of population indices of selected individual

        """

        return self.__select_individuals(num_select, elite_pop_size,              
                                         select_by='least_eval') 



    def __sort_population(self):
        """Sorts population by sort_by field (default is ascending order). 

        Notes
        -----
        - default is average fitness
        - only the population index is sorted and stored in sorted_idx_ using 
          numpy argsort.

        """
       
        try:
            self.__sorted_idx = np.argsort(self.sort_by[:self.n_pop_])
        except:
            print("Unexpected error:", sys.exc_info()[0])                           
            raise 

        if self.sort_descend:
            self.__sorted_idx = self.__sorted_idx[::-1]
        else:
            self.__sorted_idx = self.__sorted_idx                       

        return


