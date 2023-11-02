"""
Have parent clone/cell grow and create a new one.

Checking if their phenotypes diverge
"""
import numpy as np
from torch import nn
import torch
import seaborn as sns
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt

from branching_model import Agent
import importlib
importlib.reload(Agent)


def get_cell_df(cell):

    phenotypes = np.vstack(cell.pheno_changes)
    cell_time = np.arange(phenotypes.shape[0]) + cell.time_created
    pheno_df = pd.DataFrame(
        np.vstack(phenotypes),
        columns=["S", *[f"R{i}" for i in range(1, Agent.N_TREATMENTS + 1)]],
    )
    pheno_df["Time"] = cell_time
    pheno_df["ID"] = cell.id
    if cell.parent is not None:
        parent_id = cell.parent.id
    else:
        parent_id = -1

    pheno_df["Parent"] = parent_id

    return pheno_df

# def add_noise_to_weights(m, max_noise=0.25):
#     """
#     https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/5
#     """
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             m.weight.add_(torch.randn(m.weight.size()) * max_noise)


MAX_CELLS = 5
N_TIME = 200
LEARNING_RATE = 10**-1

doses = torch.zeros(Agent.N_TREATMENTS).reshape(1, -1)
# doses = torch.from_numpy(np.array([1.0, 0.0], dtype=np.float32)).reshape(
#     1, -1
# )


initial_cell = Agent.Agent(is_cell=True, id=0, learning_rate=LEARNING_RATE)
initial_cell.time_created = 0
current_max_id = initial_cell.id

cell_list = [initial_cell]
for i in range(N_TIME):
    for cell in cell_list:
        cell.update_phenotype(doses)
        p = cell.calc_growth_rate(cell.phenotype, doses)
        cell.p = p
        if hasattr(cell, "pheno_changes"):
            cell.pheno_changes.append(cell.phenotype.detach().numpy())
        else:
            cell.pheno_changes = [cell.phenotype.detach().numpy()]
        if p > 0:
            divide = np.random.binomial(1, p) == 1
            if divide:
                mutate = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                if mutate and current_max_id < MAX_CELLS - 1:
                    new_id = current_max_id + 1
                    new_cell = cell.copy(new_id=new_id)
                    new_cell.mutate()

                    new_cell.time_created = i
                    current_max_id = new_id
                    cell_list.append(new_cell)

    # elif p < 0:
    #     die = np.random.binomial(1, p) == 1
    #     if die:
    #         cell_list.remove(cell)


# cell = cell_list[1]
all_cell_df = pd.concat([get_cell_df(cell) for cell in cell_list])
plot_pheno_df = all_cell_df.melt(
    id_vars= ["Time", "ID", "Parent"], var_name="Trait", value_name="Expression"
)


plt_dst_dir = os.path.join(os.getcwd(), "tests", "mutation")
pathlib.Path(plt_dst_dir).mkdir(exist_ok=True, parents=True)
plot_f = os.path.join(plt_dst_dir, "mutation")
g = sns.FacetGrid(plot_pheno_df, col="ID", hue="Trait")
g.map(sns.lineplot, "Time", "Expression")
plt.legend()
plt.savefig(plot_f)
plt.close()
