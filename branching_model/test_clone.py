import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

from branching_model.Agent import Agent 
from branching_model.Clone import  Clone 
 Function to simulate clone growth over time with treatment parameters
def simulate_clone_growth_with_treatment(clone, n_generations, delta_drug_1, delta_drug_2, res_1, res_2):
    clone_populations = []
    for gen in range(n_generations):
        clone.grow_clones(1)  # Grow the clone population for 1 generation
        N = clone.population[-1]
        r = clone.div_rate
        new_cells = r * N
        new_mutants = clone.mut_rate * r * N
        n_killed = delta_drug_1 * N * (1 - res_1) - delta_drug_2 * N * (1 - res_2)
        new_size = N + new_cells - new_mutants - n_killed
        if new_size > clone.max_pop_size:
            new_size = clone.max_pop_size
        clone.population[-1] = new_size
        clone_populations.append(new_size)
    return clone_populations

# Function to visualize clone growth dynamics
def plot_clone_growth(clone_populations, drug_name):
    plt.plot(range(len(clone_populations)), clone_populations, label=drug_name)

div_rate = 0.5
mut_rate = 0.01
death_rate = 0.05
initial_pop_size = 100
max_pop_size = 1000
n_generations = 20

# Create an initial clone with specified parameters
initial_clone = Clone(div_rate, mut_rate, death_rate, 0, 0, initial_pop_size, initial_pop_size, max_pop_size)

# Simulate clone growth with drug_1
delta_drug_1 = 0.1  # Efficacy of drug 1
res_1 = 0.9  # Resistance to drug 1
clone_populations_1 = simulate_clone_growth_with_treatment(initial_clone, n_generations, delta_drug_1, 0.0, res_1, 1.0)
initial_clone.population[-1] = initial_pop_size  # Reset the population size

# Simulate clone growth with drug_2
delta_drug_2 = 0.2  # Efficacy of drug 2
res_2 = 0.8  # Resistance to drug 2
clone_populations_2 = simulate_clone_growth_with_treatment(initial_clone, n_generations, 0.0, delta_drug_2, 1.0, res_2)

# Visualize clone growth dynamics with sequential treatment
plt.figure()
plot_clone_growth(clone_populations_1, "Drug 1")
plot_clone_growth(clone_populations_2, "Drug 2")
plt.xlabel("Generations")
plt.ylabel("Clone Population Size")
plt.title("Sequential Treatment with Drug 1 and Drug 2")
plt.legend()
plt.show()