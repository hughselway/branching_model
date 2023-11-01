import numpy as np
from torch import nn
import torch
import seaborn as sns
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt

from branching_model import Agent, Phylogeny
import importlib
importlib.reload(Agent)
LEARNING_RATE = 10**-1


TIME_COL = "time"
SIZE_COL = "size"
MUTATION_SIZE_COL = "mutation_size"
PHENO_COL = "phenotype"
ID_COL = "id"
PARENT_ID_COL = "parent_id"
ROOT_PARENT_ID = -1


class Recorder(object):
    def __init__(self):
        self.size_df = None
        self.mutation_df = None
        self.phenotype_df = None

    def record_time_pt(self, tree, time_pt):
        sizes = {agent.id:0 for agent in tree.agents}
        mutation_sizes = {agent.id:0 for agent in tree.agents}
        phenotypes = {agent.id:None for agent in tree.agents}

        agent_ids = [agent.id for agent in tree.agents]
        edge_dict = {agent.id: agent.parent.id if agent.parent is not None else ROOT_PARENT_ID for agent in tree.agents}

        for agent in tree.agents:
            agent.phenotype
            size = 1 if agent.status == "cell" else agent.n_cells
            mutation_sizes[agent.id] += size
            sizes[agent.id] = size
            phenotypes[agent.id] = agent.phenotype.detach().numpy()

            parent = agent.parent
            while parent is not None:
                mutation_sizes[parent.id] += size
                parent = parent.parent

        phenotypes = np.vstack([phenotypes[idx] for idx in agent_ids])
        phenotype_cols = ["S", *[f"R{i}" for i in range(1, phenotypes.shape[1])]]
        _pheno_df = pd.DataFrame(phenotypes, columns=phenotype_cols)
        time_pheno_df = pd.DataFrame({TIME_COL: time_pt,
                                ID_COL:agent_ids,
                                PARENT_ID_COL:[edge_dict[idx] for idx in agent_ids]
                                }).join(_pheno_df)


        time_size_df = pd.DataFrame({TIME_COL: time_pt,
                                ID_COL:agent_ids,
                                PARENT_ID_COL:[edge_dict[idx] for idx in agent_ids],
                                SIZE_COL:[sizes[idx] for idx in agent_ids]
                                })

        time_mutation_df = pd.DataFrame({TIME_COL: time_pt,
                        ID_COL:agent_ids,
                        PARENT_ID_COL:[edge_dict[idx] for idx in agent_ids],
                        MUTATION_SIZE_COL:[mutation_sizes[idx] for idx in agent_ids]
                        })

        self.size_df = self.update_df(self.size_df, time_size_df)
        self.mutation_df = self.update_df(self.mutation_df, time_mutation_df)
        self.phenotype_df = self.update_df(self.phenotype_df, time_pheno_df)

        # return size_df, mutation_df, pheno_df
    def update_df(self, df1, df2):
        if df1 is not None and df2 is not None:
            new_df = pd.concat([df1, df2])
        elif df1 is not None and df2 is None:
            new_df = df1
        elif df1 is None and df2 is not None:
            new_df = df2
        else:
            new_df = None

        return new_df

    def write_csv(self, dst_dir, prefix=""):
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)
        size_out = os.path.join(dst_dir, f"{prefix}_{SIZE_COL}.csv").replace(f"{os.sep}_", os.sep)
        self.size_df.to_csv(size_out, index=False)

        mutation_out = os.path.join(dst_dir, f"{prefix}_{MUTATION_SIZE_COL}.csv").replace(f"{os.sep}_", os.sep)
        self.mutation_df.to_csv(mutation_out, index=False)

        pheno_out = os.path.join(dst_dir, f"{prefix}_{PHENO_COL}.csv").replace(f"{os.sep}_", os.sep)
        self.phenotype_df.to_csv(pheno_out, index=False)


if __name__ == "__main__":
    def generate_tree(n_agents=10):

        recorder = Recorder()
        tree = Phylogeny.Phylogeny()
        initial_cell = Agent.Agent(is_cell=True, id=0, learning_rate=LEARNING_RATE)
        initial_cell.time_created = 0
        current_max_id = initial_cell.id
        # cell_list = [initial_cell]
        n_created = 1
        doses = torch.zeros(Agent.N_TREATMENTS).reshape(1, -1)
        time = 0
        while n_created < n_agents:
            recorder.record_time_pt(tree, time)
            for cell in tree.agents:
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
                        if mutate:
                            new_id = current_max_id + 1
                            new_cell = cell.copy(new_id=new_id)
                            new_cell.mutate()

                            new_cell.time_created = time
                            current_max_id = new_id
                            tree.agents.append(new_cell)

                            n_created += 1
            time += 1

            # elif p < 0:
        #     die = np.random.binomial(1, p) == 1
        #     if die:
        #         cell_list.remove(cell)

        return tree, recorder


    tree, recorder = generate_tree()
    dst_dir = os.path.join(os.getcwd(), "tests/csv_files")
    recorder.write_csv(dst_dir)

