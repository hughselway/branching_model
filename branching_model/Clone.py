# import libraies
from copy import deepcopy
from networkx import ring_of_cliques
import numpy as np
import math
import itertools
from sympy import N
from torch import nn
import torch
import seaborn as sns
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from branching_model.Agent import *  
# from branching_model.Agent import Agent, CLONE_MUTATION_RATE 

class Clone(Agent):
    def __init__(
        self,
        div_rate: float,
        mut_rate: float,
        death_rate: float,
        new_mutants: int = None, 
        n_killed: int = None,  
        new_size: int = None,
        initial_pop_size: int = None,
        max_pop_size: int = None,
        # population: int = None
        # initial_fitness = None 
    ):
        super().__init__(is_cell=False, id=None)
        
        self.div_rate = div_rate
        self.mut_rate = mut_rate
        self.death_rate = death_rate
        self.new_mutants = new_mutants
        self.n_killed = n_killed
        self.new_size = new_size
        self.initial_pop_size = initial_pop_size
        self.max_pop_size = max_pop_size
        self.population = [initial_pop_size]

    def calculate_new_size(self, new_id, r, N, delta_drug_1, res_1, delta_drug_2, res_2): # new variables
        new_clones = Clone(self.div_rate, self.mut_rate, self.death_rate, 0, 0, self.new_size, self.initial_pop_size, self.max_pop_size) # from Clone class
        new_clones = Agent.copy()
        div_rate = self.div_rate
        N = self.initial_pop_size #re-assign initial_pop_size to N (for formula use)
        r = div_rate #re-assign div_rate to r (for formula use)
        new_cells = r * N
        mut_rate = CLONE_MUTATION_RATE
        new_mutants = self.mut_rate * r * N
        n_killed = delta_drug_1 * N * (1 - res_1) - delta_drug_2 * N * (1 - res_2)
        new_size = N + new_cells - new_mutants - n_killed
        
        new_clones.div_rate = div_rate
        new_clones.new_size = new_size
        return new_clones

    def create_mutant(self, div_rate, initial_pop_size, new_id):
        r = div_rate
        N = initial_pop_size
        mut_rate = CLONE_MUTATION_RATE
        delta_drug_1 = 0.1 # efficacy of drug 1
        res_1 = 0.9 # resistance to drug 1
        delta_drug_2 = 0.2 # efficacy of drug 2
        res_2 = 0.8 # resistance to drug 2

        new_clone = Clone(div_rate, mut_rate, 0.0, 0.0, initial_pop_size)
        
        # Clone and update the attributes
        return new_clone.clone(new_id, r, N, delta_drug_1, res_1, delta_drug_2, res_2)