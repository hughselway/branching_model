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
LEARNING_RATE = 10**-1


TIME_COL = "time"
SIZE_COL = "size"
MUTATION_SIZE_COL = "mutation_size"
PHENO_COL = "phenotype"
ID_COL = "id"
CLONE_ID_COL = "clone_id"
PARENT_ID_COL = "parent_id"
VAF_COL = "vaf"
ROOT_PARENT_ID = -1

CLONE_STR = "clone"
CELL_STR = "cell"

class Recorder(object):
    def __init__(self):
        self.clone_mutation_df = None
        self.cell_mutation_df = None
        self.size_df = None

    def get_id(self, agent, resolution=CLONE_STR):
        if resolution == CLONE_STR:
            mut_id = agent.clone_id
        else:
            mut_id = agent.id
        return mut_id

    def get_mutation_df(self, agent_list, time_pt, resolution=CLONE_STR):
        # resolution = CELL_STR
        all_ids = [a.id for a in agent_list]
        id_dict = {a.id: self.get_id(a, resolution=resolution) for a in agent_list}
        root_idx = np.argmin(all_ids)
        root_agent = agent_list[root_idx]
        root_agent_id = root_agent.id
        edge_dict = {root_agent_id: ROOT_PARENT_ID}  # child : parent

        mutant_ids = np.unique(list(id_dict.values()))
        mutant_ids = sorted(mutant_ids)
        clone_clusters = {i: [] for i in mutant_ids}
        for agent in agent_list:
            mut_id = id_dict[agent.id]
            clone_clusters[mut_id].append(agent)
            parent = agent.parent
            while parent is not None:
                assert parent != agent
                parent_mut_id = id_dict[parent.id]
                clone_clusters[parent_mut_id].append(agent)
                if parent_mut_id != mut_id:
                    if mut_id not in edge_dict:
                        edge_dict[mut_id] = parent_mut_id
                parent = parent.parent

        # No time to figure out why clones end up in the same clone list multiple times :(
        filtered_clone_clusters = {idx: set(clone_clusters[idx]) for idx in clone_clusters}
        mutation_sizes = {idx: sum([1 if a.status == CELL_STR else a.num_cells for a in filtered_clone_clusters[idx]]) for idx in mutant_ids}
        phenotypes = {idx: np.vstack([a.phenotype.detach().numpy() for a in filtered_clone_clusters[idx]]).mean(axis=0)  for idx in mutant_ids}
        n_pheno = len(phenotypes[0])
        phenotype_cols = ["S", *[f"R{i}" for i in range(1, n_pheno)]]

        phenotype_df = pd.DataFrame([phenotypes[idx] for idx in mutant_ids], columns=phenotype_cols)
        phenotype_df[CLONE_ID_COL] = mutant_ids

        mutation_df = pd.DataFrame(
            {
                TIME_COL: time_pt,
                CLONE_ID_COL: mutant_ids,
                PARENT_ID_COL: [edge_dict[idx] for idx in mutant_ids],
                MUTATION_SIZE_COL: [mutation_sizes[idx] for idx in mutant_ids]
            }

        )

        mutation_df= mutation_df.merge(phenotype_df, on=CLONE_ID_COL)

        return mutation_df


    def record_time_pt(self, agent_list, time_pt):
        sizes = {agent.id: 0 for agent in agent_list}
        phenotypes = {agent.id: None for agent in agent_list}
        clone_ids = {agent.id: agent.clone_id for agent in agent_list}

        agent_ids = [agent.id for agent in agent_list]

        edge_dict = {
            agent.id: agent.parent.id if agent.parent is not None else ROOT_PARENT_ID
            for agent in agent_list
        }

        for agent in agent_list:
            size = 1 if agent.status == "cell" else agent.n_cells
            # mutation_sizes[agent.id] += size
            sizes[agent.id] = size
            phenotypes[agent.id] = agent.phenotype.detach().numpy()


        phenotypes = np.vstack([phenotypes[idx] for idx in agent_ids])
        phenotype_cols = ["S", *[f"R{i}" for i in range(1, phenotypes.shape[1])]]
        _pheno_df = pd.DataFrame(phenotypes, columns=phenotype_cols)

        time_pt_size_df = pd.DataFrame(
            {
                TIME_COL: time_pt,
                ID_COL: agent_ids,
                CLONE_ID_COL: [clone_ids[idx] for idx in agent_ids],
                PARENT_ID_COL: [edge_dict[idx] for idx in agent_ids],
                SIZE_COL: [sizes[idx] for idx in agent_ids]
            }
        ).join(_pheno_df)

        time_pt_clone_mutation_df = self.get_mutation_df(agent_list=agent_list, time_pt=time_pt, resolution=CLONE_STR)
        clone_res_vaf = self.calculate_vaf(time_pt_size_df, time_pt_clone_mutation_df)
        vaf_col_pos = list(time_pt_clone_mutation_df).index(MUTATION_SIZE_COL) + 1
        time_pt_clone_mutation_df.insert(loc=vaf_col_pos, column=VAF_COL, value=clone_res_vaf)

        self.size_df = self.update_df(self.size_df, time_pt_size_df)
        self.clone_mutation_df = self.update_df(self.clone_mutation_df, time_pt_clone_mutation_df)

        if agent_list[0].status == CELL_STR:
            time_pt_cell_mutation_df = self.get_mutation_df(agent_list=agent_list, time_pt=time_pt, resolution=CELL_STR)
            cell_res_vaf = self.calculate_vaf(time_pt_size_df, time_pt_cell_mutation_df)
            vaf_col_pos = list(time_pt_cell_mutation_df).index(MUTATION_SIZE_COL) + 1
            time_pt_cell_mutation_df.insert(loc=vaf_col_pos, column=VAF_COL, value=cell_res_vaf)
            self.cell_mutation_df = self.update_df(self.cell_mutation_df, time_pt_cell_mutation_df)

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

        size_fout = os.path.join(dst_dir, f"{prefix}_size.csv").replace(f"{os.sep}_", os.sep)
        self.size_df.to_csv(size_fout, index=False)

        clone_mutaton_fout = os.path.join(dst_dir, f"{prefix}_clone_mutations.csv").replace(f"{os.sep}_", os.sep)
        self.clone_mutation_df.to_csv(clone_mutaton_fout, index=False)

        cell_mutaton_fout = os.path.join(dst_dir, f"{prefix}_cell_mutations.csv").replace(f"{os.sep}_", os.sep)
        self.cell_mutation_df.to_csv(cell_mutaton_fout, index=False)

    def long_to_wide(self, cname):
        info_cols = [ID_COL, PARENT_ID_COL]
        info_df = self.df[info_cols].drop_duplicates()

        long_df = self.df[[ID_COL, TIME_COL, cname]]
        wide_df = long_df.pivot(index=ID_COL, columns=TIME_COL, values=cname)
        wide_df.fillna(0, inplace=True)
        wide_df = info_df.merge(wide_df, on=ID_COL)

        return wide_df

    def calculate_vaf(self, size_df, mutation_df):
        """
        resolution : str
            cell or clone
        """

        n_cells = sum(size_df[SIZE_COL])
        n_seq = 2*n_cells
        vaf = mutation_df[MUTATION_SIZE_COL]/n_seq
        return vaf



if __name__ == "__main__":

    def generate_tree(n_agents=10):
        recorder = Recorder()
        self = recorder
        agent_list = []
        initial_cell = Agent.Agent(is_cell=True, id=0, clone_id=0, learning_rate=LEARNING_RATE)
        initial_cell.time_created = 0
        agent_list.append(initial_cell)
        # current_max_id = initial_cell.id
        n_created = 1
        n_clones = 1
        doses = torch.zeros(Agent.N_TREATMENTS).reshape(1, -1)
        time = 0
        while n_clones < n_agents:
            recorder.record_time_pt(agent_list, time)
            for cell in agent_list:
                cell.update_phenotype(doses)
                p = cell.calc_growth_rate(cell.phenotype, doses, resistance_cost=0.2, resistance_benefit=0.5)
                cell.p = p
                if hasattr(cell, "pheno_changes"):
                    cell.pheno_changes.append(cell.phenotype.detach().numpy())
                else:
                    cell.pheno_changes = [cell.phenotype.detach().numpy()]
                if p > 0:
                    divide = np.random.binomial(1, p) == 1
                    if divide:
                        mutate_cell = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                        if mutate_cell:
                            n_created += 1
                            mutate_clone = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                            if mutate_clone:
                                n_clones += 1

                            new_cell = cell.copy(new_id=n_created-1, new_clone_id=n_clones-1)
                            new_cell.mutate()

                            new_cell.time_created = time
                            agent_list.append(new_cell)

            time += 1

        return recorder, agent_list

    recorder, agent_list = generate_tree()
    self = recorder
    dst_dir = os.path.join(os.getcwd(), "tests/csv_files")
    recorder.write_csv(dst_dir)
