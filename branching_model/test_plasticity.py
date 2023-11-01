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

def test_adaptation(doses, plot_title, plot_f, n_steps=100, learning_rate=10**-1):
    cell = Agent.Agent(id=0, is_cell=True, learning_rate=learning_rate)
    phenotypes = [None] * n_steps
    for i in range(n_steps):
        cell.update_phenotype(doses)
        phenotypes[i] = cell.phenotype.detach().numpy()

    phenotypes = np.vstack(phenotypes)
    # phenotypes = 1 - phenotypes
    time_array = np.arange(0, n_steps)
    pheno_df = pd.DataFrame(
        np.vstack(phenotypes),
        columns=["S", *[f"R{i}" for i in range(1, Agent.N_TREATMENTS + 1)]],
    )
    pheno_df["Time"] = time_array
    plot_pheno_df = pheno_df.melt(
        id_vars="Time", var_name="Trait", value_name="Expression"
    )

    sns.lineplot(plot_pheno_df, x="Time", y="Expression", hue="Trait")
    plt.title(plot_title)
    if plot_f is not None:
        plt_dst_dir = os.path.split(plot_f)[0]
        pathlib.Path(plt_dst_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_f)
        plt.close()
    else:
        plt.show()


def test_treatment(
    plot_title: str, plot_f: str, n_steps: int = 100, learning_rate: float = 10**-1
):
    cell = Agent.Agent(is_cell=True, id=0, learning_rate=learning_rate)
    phenotypes = [None] * 3 * n_steps
    no_treatment_doses = torch.zeros(Agent.N_TREATMENTS).reshape(1, -1)
    for i in range(n_steps):
        cell.update_phenotype(no_treatment_doses)
        phenotypes[i] = cell.phenotype.detach().numpy()

    treatment1_doses = torch.from_numpy(np.array([1.0, 0.0], dtype=np.float32)).reshape(
        1, -1
    )
    for i in range(n_steps):
        cell.update_phenotype(treatment1_doses)
        phenotypes[i + n_steps] = cell.phenotype.detach().numpy()

    treatment2_doses = torch.from_numpy(np.array([0.0, 1.0], dtype=np.float32)).reshape(
        1, -1
    )
    for i in range(n_steps):
        cell.update_phenotype(treatment2_doses)
        phenotypes[i + 2 * n_steps] = cell.phenotype.detach().numpy()

    phenotypes = np.vstack(phenotypes)
    time_array = np.arange(0, phenotypes.shape[0])
    pheno_df = pd.DataFrame(
        phenotypes, columns=["S", *[f"R{i}" for i in range(1, Agent.N_TREATMENTS + 1)]]
    )
    pheno_df["Time"] = time_array
    plot_pheno_df = pheno_df.melt(
        id_vars="Time", var_name="Trait", value_name="Expression"
    )

    sns.lineplot(plot_pheno_df, x="Time", y="Expression", hue="Trait")
    ymax = np.max(phenotypes)
    plt.title(plot_title)
    plt.vlines(n_steps, 0, ymax, color="black", linestyles="--")
    plt.vlines(2 * n_steps, 0, ymax, color="black", linestyles="--")
    if plot_f is not None:
        plt_dst_dir = os.path.split(plot_f)[0]
        pathlib.Path(plt_dst_dir).mkdir(exist_ok=True, parents=True)
        plt.savefig(plot_f)
        plt.close()
    else:
        plt.show()


plt_dst_dir = os.path.join(os.getcwd(), "tests", "cell_plasticity")

no_treament_f = os.path.join(plt_dst_dir, "no_treatment.png")
no_treament_title = "No treatment"
no_treatment_doses = torch.zeros(Agent.N_TREATMENTS).reshape(1, -1)
test_adaptation(
    doses=no_treatment_doses, plot_title=no_treament_title, plot_f=no_treament_f
)


treament1_f = os.path.join(plt_dst_dir, "treatment_1.png")
treament1_title = "Treatment 1"
treatment1_doses = torch.from_numpy(np.array([1.0, 0.0], dtype=np.float32)).reshape(
    1, -1
)
test_adaptation(doses=treatment1_doses, plot_title=treament1_title, plot_f=treament1_f)

treament2_f = os.path.join(plt_dst_dir, "treatment_2.png")
treament2_title = "Treatment 2"
treatment2_doses = torch.from_numpy(np.array([0.0, 1.0], dtype=np.float32)).reshape(
    1, -1
)
test_adaptation(doses=treatment2_doses, plot_title=treament2_title, plot_f=treament2_f)

dual_f = os.path.join(plt_dst_dir, "treatment_dual.png")
dual_treament_title = "Dual treatments"
dual_doses = torch.from_numpy(np.array([1.0, 1.0], dtype=np.float32)).reshape(1, -1)
test_adaptation(doses=dual_doses, plot_title=dual_treament_title, plot_f=dual_f)

sequential_treatment_f = os.path.join(plt_dst_dir, "sequential_treatment.png")
test_treatment("Sequential", sequential_treatment_f)
