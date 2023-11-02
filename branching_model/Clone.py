# import libraies
from copy import deepcopy
from networkx import ring_of_cliques
import numpy as np
import math
import itertools
import sympy as sp
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
        initial_pop_size: int = 1,
        max_pop_size: int = None,
        # initial_fitness = None 
    ):
        super().__init__(is_cell=False, id=None)
        
        self.div_rate = div_rate
        self.mut_rate = mut_rate
        self.death_rate = death_rate
        self.new_mutants = new_mutants
        self.n_killed = n_killed
        self.max_pop_size = max_pop_size
        self.population = [initial_pop_size]

    def calculate_new_clone_size(self, delta_drug_1, resistance_1, delta_drug_2, resistance_2): # new variables
        new_cells = self.div_rate * self.n_cells
        new_mutants = new_cells * CLONE_MUTATION_RATE  # new cells that are dividing and mutating
        n_killed = delta_drug_1 * self.n_cells * (1 - resistance_1) - delta_drug_2 * self.n_cells * (1 - resistance_2) # cells that are dying
        new_clone_size = self.n_cells + new_cells - new_mutants - n_killed
        
        return new_clone_size, new_mutants