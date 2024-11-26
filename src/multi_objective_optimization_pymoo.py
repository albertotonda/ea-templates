# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 08:03:50 2024

Script with multi-objective evolutionary optimization, centered around the 
pymoo library.

@author: Alberto Tonda
"""
import datetime
import json
import numpy as np
import os
import pymoo
import types # only used for a weird thing

from pymoo.core.callback import Callback
from pymoo.core.problem import Problem


# local scripts
from common_logging import initialize_logging, close_logging

def prepare_output_folder(output_folder : str) -> str :
    """
    Prepare directory name for the output folder, prepending the current date.
    """
    
    # find base name of the folder (at the end of the path)
    folder_name = os.path.basename(output_folder)
    folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-" + folder_name
    
    output_folder = os.path.join(os.path.dirname(output_folder), folder_name)
    
    return output_folder

class ExampleProblem(Problem):
    """
    This is just an example, implementing the ZDT1 test function. However, it
    also accepts extra arguments in the constructor, and additional arguments
    can be added if necessary for your problem. It is also pre-disposed for 
    using multi-thread or multi-process parallel computations.
    """
    logger = None
    process_pool = None
    thread_pool = None
    
    def __init__(self, logger=None, process_pool=None, thread_pool=None) -> None :
        super().__init__(n_var=30, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        self.logger = logger

    def _evaluate(self, x, out, *args, **kwargs) :
        """
        The _evaluate function is expecting 'x' to be a numpy matrix of candidate
        solutions, with each row a different candidate solution, and each column
        a different variable. The return value is expected to be another numpy
        matrix, with the fitness value of each candidate solution on each row,
        and each column being a different fitness function.
        
        The only complicated part here is that, if multiprocessing or multithread
        has been initialized, we need to manage that.
        """
        # TODO manage multiprocessing or multithreading here
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis=1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out["F"] = np.column_stack([f1, f2])

class ExampleCallback(Callback) :
    """
    This class implements a callback function that is invoked at the end of each
    iteration (generation) of the multi-objective evolutionary algorithm.
    """
    
    def __init__(self, population_output_file_name="population") -> None :
        super().__init__()
        self.population_output_file_name = population_output_file_name
        
    def notify(self, algorithm) :
        """
        This is the method that is automatically called at the end of each
        generation.
        """

if __name__ == "__main__" :
    
    # potentially read all necessary hyperparameters from a JSON file
    configuration_file = "../examples/multi_objective_optimization_pymoo.json"
    configuration_dictionary = dict()
    
    # if the configuration_file has been specified, read it
    if 'configuration_file' in locals() :
        configuration_dictionary = json.load(open(configuration_file, "r"))
    
    # or specify hyperparameters by hand
    seed = configuration_dictionary.get("seed", 42)
    population_size = configuration_dictionary.get("population_size", 100)
    max_generations = configuration_dictionary.get("max_generations", 50)
    output_folder = configuration_dictionary.get("output_folder", "../local/multi-objective-optimization-pymoo")
    restart_from_checkpoint = configuration_dictionary.get("restart_from_checkpoint", None)
    population_seeds = configuration_dictionary.get("population_seeds", None)
    
    # take all variables in locals() and store them in the configuration_dictionary
    local_variables = locals().copy()
    for key, value in local_variables.items() :
        if not key.startswith("_") and not callable(value) and \
            not isinstance(value, type) and not isinstance(value, types.ModuleType) and \
                key != "configuration_dictionary" :
                    configuration_dictionary[key] = value
    
    # prepare and seed a numpy random number generator object, which might be
    # used later, for example to initialize the starting population
    prng = np.random.default_rng(seed=seed)
    
    # prepare folder name including the date
    output_folder = prepare_output_folder(output_folder)
    if not os.path.exists(output_folder) :
        os.makedirs(output_folder)
        
    # save the configuration file inside the output folder
    with open(os.path.join(output_folder, "configuration.json"), "w") as fp :
        json.dump(configuration_dictionary, fp, indent=4)
        
    # initialize logging
    logger = initialize_logging(output_folder, "moea-pymoo-log")
    
    # print information in the log
    logger.info("Experiment using pymoo with output in folder \"%s\": random seed %d, population_size=%d, max_generations=%d" 
                % (output_folder, seed, population_size, max_generations))
    
    # set up the instance of the Problem that we are going to tackle
    problem = ExampleProblem(logger=logger)
    logger.info("Problem class: %s" % problem.__class__.__name__)
    
    # there are also another couple of classes to be instantiated, mostly for
    # utility (like, saving the population at every iteration)
    
    # before exiting, close all handlers of the logger
    close_logging(logger)