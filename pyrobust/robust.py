"""Robust optimisation class.

PYTHON version based on source code from University of Exeter
J. E. Fieldsend, K. Alyahya, K. Doherty

see pop_project/src/GECCO_2017/EAS_GA_UpdateHistory

Authors: N. Banglawala, EPCC, 2019
License: MIT

"""

import sys

import numpy as np

from pyrobust import population
from pyrobust import statistical_utils as su
from pyrobust import resampling_methods as rm
from pyrobust import weighting_methods as wm


###############################################################################
# DICTIONARIES AND DEFAULTS

# Resampling method options
RESAMPLING=dict(uniform=rm.uniform_resampling, sobol=rm.sobol_resampling,
                wasserstein=rm.wasserstein_resampling,
                voronoi_mc=rm.voronoi_monte_carlo_resampling)

resampling_default = 'wasserstein'
# default rs_sample_size = 3**dims

# Weighting method options
WEIGHTING=dict(uniform=wm.uniform_weights, wasserstein=wm.wasserstein_weights,
              voronoi_mc=wm.voronoi_monte_carlo_weights)

weighting_default = 'wasserstein'


SAMPLING=dict(uniform=np.random.uniform, sobol=su.generate_sobol_sample,
              latin=su.generate_latin_sample)

sampling_default = 'latin'


###############################################################################

class Robust(object):
    """

    Parameters
    ----------

    max_pop : int
        maximum population size

    problem : class instance
        contains problems parameters and methods, such as fitness


    # OPTIONS
    use_history : bool, default: False 
        use search history to improve individual's robust fitness
        estimate using search history of neighbours with set defaults

    use_history_with : {}, default: None
        advanced options for specifying function (func) to improve robust
        fitness estimate, sampling and weighting methods. Specify options in
        dictionary:

        {'func' : func, 'sampling' : func, 'weighting': func}

        Note: if key not set or value is None, defaults used. 

    update_history : bool, default: False
        improve robust fitness estimate of all neighbours of individual
        using individual's fitness with set defaults

    update_history_with : {}, default: None
        advanced options for specifying function (func) to improve robust
        fitness estimate of neighbours of individual. Specify options in 
        dictionary:  
        
        {'func' : func, 'sampling' : func, 'weighting': func}
            
        Note: if key not set or value is None, defaults used.
 
    resample : {'uniform', 'sobol', 'wasserstein'}, default: None
        resample individual with least fitness evaluations. This could be from
        the entire population or from elites only depending on options set.

    resample_with : {'resampling': func, 'weighting': func,
                     'sample_size' : int},
                    defaults: {}
            resampling : resampling function
            weighting : weighting function to use in resampling
            sample_size : size of samples used to generate candidate points

            defaults: {}
        advanced options for resampling. Note: when non-empty dict, if key not
        set or value is None, defaults used.

    maximise : bool, default: False 
        maximise objective function
  

    Attributes
    ----------

    tot_evals : int
        total number of fitness evaluations

    """

    def __init__(self, max_pop, problem, use_history=False, 
                 use_history_with=None, update_history=False, 
                 update_history_with=None, resample=None, resample_with=None, 
                 maximise=False):

        self.pop = population.Population(max_pop, problem, 
                                         sort_descend=maximise)
        self.max_pop = max_pop
        self.problem = problem
        self.tot_evals = 0

        self.lbound, self.ubound = problem.get_bounds()
        self.dlbound, self.dubound = problem.get_disturbance_bounds()
        self.dims = len(self.dlbound)

        # set options for improving robust fitness estimate
        self.__set_history_options(use_history, use_history_with,
                                   update_history, update_history_with)

        # index list of individuals to ignore
        self.__exclude = []


        # once options for using history are set, set wrappers
        self.pop.add_individual = self.add_individual # individual
        self.problem.calc_fitness = self.calc_fitness # neighbours

        # set options for resampling
        self.__set_resampling_options(resample, resample_with)


    def get_tot_evals(self):
        return self.tot_evals


    def update_tot_evals(self):
        self.tot_evals = self.tot_evals + 1
        return


    def reset(self):
        """Reset robust optimisation to empty state."""

        self.pop.reset()
        self.tot_evals = 0


    def get_best(self, num):
        """Returns num number of best solutions and their estimated robust
           fitness."""

        x, xfit, xrob_fit = self.pop.get_fittest(num)

        return x, xrob_fit


    def create_initial_population(self, n_init, init_sampling='uniform'):
        """Creates initial population from random sample.

        Parameters
        ----------
        n_init : int
            size of initial population

        init_sampling : {'uniform', 'latin', 'sobol',...}, default: 'uniform'
            random sampling method to use to generate initial population

        """

        try:
            sample_shape = (n_init, self.dims)
            init_sample = su.generate_random_sample(self.problem.lbound,
                                                    self.problem.ubound,
                                                    sample_shape,
                                                    sampling=init_sampling)

            self.pop._create_initial_population(init_sample)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise


    def run(self, max_evals):
        """Run robust optimisation for max_evals evaluations where each
        evaluation is an evaluation of the objective (fitness) function.

        Parameters
        ----------
        max_evals : int
            maximum number of evaluations
 
        """

        try:
            while self.tot_evals < max_evals:
                self.iterate()

            return

        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise
        


    def iterate(self):
        """Main iteration of robust optimisation."""

        # step 1 : create new individual and add to population
        # Note: if improving robust fitness estimate using search history, this 
        # will get done when calculating fitness (for improve neighbours) and 
        # when adding individual to population (for improve individual) 

        self.pop.evolve_and_add_to_population()

        # step 2 : resample elite sub-population
        self.resampling_call(self.rs_select_func, self.resampling_func, 
                        self.rs_weighting_func, self.dlbound, self.dubound, 
                        self.problem.elite_pop_size, self.rs_sample_size)         

        return


    def __set_history_options(self, use_history, use_history_with,
                              update_history, update_history_with):
        """Set options for improving robust fitness estimate using search
        history."""

        # set defaults
        use_hist_func = lambda *args, **kwargs: None
        use_hist_samp = lambda *args, **kwargs: None
        use_hist_weight = lambda *args, **kwargs: None

        update_hist_func = lambda *args, **kwargs: None
        update_hist_samp  = lambda *args, **kwargs: None
        update_hist_weight = lambda *args, **kwargs: None

        try:
            if use_history_with:
                # use history to improve robust fitness of individuals
                # advanced options override predefined options

                # set use_history function
                if 'func' in use_history_with:
                    use_hist_func = use_history_with['func']
                else:
                    # default
                    use_hist_func = self.use_history

                # set sampling/weighting methods
                if 'sampling' in use_history_with:
                    use_hist_samp = use_history_with['sampling']
                else:
                    # default
                    use_hist_samp  = SAMPLING[sampling_default]

                if 'weighting' in use_history_with:
                    use_hist_weight = use_history_with['weighting']
                else:
                    # default
                    use_hist_weight  = WEIGHTING[weighting_default]
            elif use_history:
                # else set predefined options
                use_hist_func = self.use_history                                                        
                use_hist_samp = SAMPLING[sampling_default]                   
                use_hist_weight = WEIGHTING[weighting_default]   


            if update_history_with:
                # use history to improve fitness for neighbours
                # advanced options override predefined options

                # set update_history function
                if 'func' in update_history_with:
                    update_hist_func = update_history_with['func']
                else:
                    # default
                    update_hist_func = self.update_history

                # set sampling/weighting methods
                if 'sampling' in update_history_with:
                    update_hist_samp = update_history_with['sampling']
                else:
                    # default
                    update_hist_samp  = SAMPLING[sampling_default]

                if 'weighting' in update_history_with:
                    update_hist_weight = update_history_with['weighting']
                else:
                    # default
                    update_hist_weight = WEIGHTING[weighting_default]

            elif update_history:
                # else set predefined options
                update_hist_func = self.update_history                 
                update_hist_samp = SAMPLING[sampling_default]                       
                update_hist_weight = WEIGHTING[weighting_default] 


            # set options as attributes
            setattr(self, 'use_hist_func', use_hist_func)
            setattr(self, 'use_hist_samp', use_hist_samp)
            setattr(self, 'use_hist_weight', use_hist_weight)
            setattr(self, 'update_hist_func', update_hist_func)
            setattr(self, 'update_hist_samp', update_hist_samp)
            setattr(self, 'update_hist_weight', update_hist_weight)

        except KeyError as err:
            print("Method used in improving fitness unknown, got ", err)
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise



    def __set_resampling_options(self, resample, resample_with):
        """Set options for resampling."""

        # set defaults
        resampling_func     = lambda *args, **params: None
        rs_weighting_func = lambda *args, **params: None
        rs_sample_size    = 3**(self.dims)

        # set resampling call
        resampling_call = self.__resampling

        rs_select_func = self.pop.select_least_evaluated  

        try:
            if resample_with:
                # advanced options override predefined resampling
                if 'resampling' in resample_with:
                    resampling_func = \
                        RESAMPLING[resample_with['resampling']]
                else:
                    resampling_func = RESAMPLING[resampling_default]

                if 'weighting' in resample_with:
                    rs_weighting_func = \
                        WEIGHTING[self.resample_with['weighting']]
                else:
                    # default
                    rs_weighting_func = _WEIGHTING[weighting_default]

                if 'sample_size' in resample_with:
                    rs_sample_size = resample_with['sample_size']
   
                # set resampling call
                resampling_call = self.__resampling
 
            elif resample:
                # predefine resampling : note resample and weighting functions
                # are from same method

                resampling_func = RESAMPLING[resample]
                rs_weighting_func = WEIGHTING[resample]

            else:    
                # set resampling call to None                                     
                resampling_call = lambda *args, **params: None

            # set options as attributes
            setattr(self, 'resampling_func', resampling_func)
            setattr(self, 'rs_weighting_func', rs_weighting_func)
            setattr(self, 'rs_sample_size', rs_sample_size)
            setattr(self, 'resampling_call', resampling_call)

            setattr(self, 'rs_select_func', rs_select_func)

            return

        except KeyError as err:
            print("Method used in resampling unknown, got ", err)
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise


    def add_individual(self, x, x_fit=None, x_rob_fit=None, *args,  **params):
        """Wrapper for individual's robust fitness, for after adding
        individuial."""

        # add individual to population
        x_idx, x_fit, x_rob_fit = \
            self.pop._add_individual(x, x_fit=x_fit, x_est_fit=x_rob_fit,
                                     *args, **params)

        # use search history to individual's improve robust fitness
        self.use_hist_func([x_idx], self.use_hist_samp, 
                                  self.use_hist_weight, *args, **params)

        return x_idx, x_fit, x_rob_fit


    def calc_fitness(self, x, *args, **params):
        """Wrapper for updating robust fitness estimate of neighbours."""

        # calculate fitness
        x_fit = self.problem.fitness(x, *args, **params)

        # update total evaluations
        self.update_tot_evals()

        # improve fitness of neighbours using x_fit value
        self.update_hist_func(x, x_fit, self.update_hist_samp, 
                            self.update_hist_weight, *args, **params)

        return x_fit


    def __resampling(self, select_func, resampling_func, weighting_func,
                     dlbound, dubound, max_elite_pop, sample_size, 
                     add_new=True):
        """Resamples selected individual using selection, sampling and
        weighting methods and updates the robust fitness estimats of the
        individual.

        Parameters
        ----------
        select_func : func
            method to select point to resample from elites

        resampling_func : func
            method to generate resampled point

        weighting_func : func
            method to generate weights

        dlbound, dubound : float, float
            bounds of disturbance neighbourhood

        max_elite_pop : int
            maximum size of elite population from which to select point to
            resample

        sample_size : int
            number of candidate points to generate

        add_new : bool, default: True
            add valid new resampled point if True

        """

        # number of individuals to resample
        num_select = 1

        # get elite individual to resample
        x_idx = select_func(num_select, max_elite_pop)[0]

        x = self.pop.get_individual(x_idx)

        # get evaluation points in x's disturbance neighbourhood
        # (i.e. its neighbours)
        x_nbrs = self.pop.get_neighbour(x_idx, None)

        # get new x and weights by resampling disturbance neighbourhood of x
        x_new, weights, cpts = self.resampling_func(x, x_nbrs, dlbound, 
                                                    dubound, sample_size,
                                                    weighting=weighting_func)

        # exclude x from improve fitness of neighbours if it set as an option..
        self.__exclude = [x_idx]

        # evalute fitness of new x
        x_new_fit = self.problem.calc_fitness(x_new)

        self.__exclude = []

        # add new neighbour and fitness value to x
        self.pop.add_neighbour(x_idx, x_new, x_new_fit)

        # weighted sum
        x_new_rob_fit = \
            np.sum(self.pop.get_fitness_value(x_idx, None) * weights)

        # update x's robust fitness
        self.pop.set_fitness_estimate(x_idx, x_new_rob_fit)        

        # add x_new to population if x_new within legal bounds
        if add_new:
            if su._is_in_bounds(x_new, self.lbound, self.ubound):
                xidx, xfit, xrobfit = self.pop._add_individual(x_new, x_new_fit, x_new_fit)

        return


    def __use_history(self, x_idx, pts, p_fit, sampling, weighting):
        """Use search history to update robust fitness estimate of individual
        or neighbours of given individual.

        Parameters
        ----------
         x_idx : list or ndarray, dtype=int
            population indices of individuals to update.

        pts : ndarray, dtype=float
            points to consider

        p_fit : ndarray, dtype=float
            fitness values of points being considered

        sampling : func
            sampling method to use for candidate points

        weighting : func
            weighting method for final weighted sum of robust fitness

        """

        try:
            
            # check and sanitise indices and points array
            # check if 1d for index list / array
            x_idx = self.pop._check_dims(x_idx, 1)

            # note: not the same as dimensions of the problem!
            pts = self.pop._check_dims(pts, 2)

            p_fit = self.pop._check_dims(p_fit, 1)

            # generate sample for candidates
            sample_shape =  (3**self.dims, self.dims)

            sample = sampling(0, 1, sample_shape)

            # scale and shift samples
            sample = sample * (self.dubound - self.dlbound) + self.dlbound

            for i in x_idx:
                x = self.pop.get_individual(i)

                # track if at least one neighbour found, then must update
                to_update = False

                for j, p in enumerate(pts):

                    if np.all(abs(x - p) <= self.dubound): #and \
                        to_update = True

                        self.pop.add_neighbour(i, p, p_fit[j])

                # update objective average
                # if individual has had at least one neighbour added
                if to_update:

                    # create new (disturbed) x
                    candidates = x + sample

                    # evaluation points consist of x and points in its
                    # disturbance neighbourhood
                    x_eval_set = \
                        np.vstack([x, self.pop.get_neighbour(i, None)])

                    weights = \
                        weighting(x_eval_set, candidates, self.dims)

                    fit_vals = self.pop.get_fitness_value(i, None)
                    weighted_fitness = fit_vals * weights

                    # update objective average
                    self.pop.set_fitness_estimate(i, np.sum(weighted_fitness))
 
            return

        except:
            print("Unexpected exception", sys.exc_info()[0])
            raise


    def use_history(self, x_idx, sampling, weighting, **params):
        """Improves robust fitness estimate of individual using all neighbours
        in population.

        Parameters
        ----------
        x_idx : list or ndarray, dtype=int
            population index of individual(s) to update

        sampling : func
            sampling method to use for candidate points

        weighting : func
            weighting method for final weighted sum of robust fitness

        Notes
        -----
        It is assumed that the individual to add is part of the population

        """

        # consider entire poplation as potential neighbours, excluding x_idx
        no_i_idx = np.delete(np.arange(self.pop.get_pop_size()), x_idx) 

        # exclude listed individuals
        for idx in self.__exclude:
                no_i_idx = np.delete(no_i_idx, idx)

        pts = self.pop.get_individual(no_i_idx)

        # fitness value of all potential neighbours
        p_fit = self.pop.get_fitness_value(no_i_idx, 0)

        #self.__improve_robust_fitness
        self.__use_history(x_idx, pts, p_fit, sampling, weighting)

        return


    def update_history(self, x, x_fit, sampling, weighting, **params):
        """Improves robust fitness estimate of all neighbours using search
        history of an individual.

        Parameters
        ----------
        x  : ndarray, dtype=float
            individual to use for updating robust fitness of neighbours

        x_fit : float
            x's fitness value

        sampling : func
            sampling method to use for candidate points

        weighting : func
            weighting method for final weighted sum of robust fitness

        """

        # points to update
        pts_idx = np.arange(self.pop.get_pop_size())

        # exclude these individuals
        for idx in self.__exclude:                                          
            pts_idx = np.delete(pts_idx, idx)  

        self.__use_history(pts_idx, x, x_fit, sampling, weighting)

        return


