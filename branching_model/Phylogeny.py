import numpy as np
import numpy.random as npr
import os
import torch
from torch import nn

from branching_model.Agent import Agent


class Phylogeny(object):
    def __init__(
        self,
        is_cell: bool = True,
        learning_rate: float = 1 * (10**-3),
        optimizer_cls: type = torch.optim.SGD,
        activation_fxn: nn.Module = nn.ReLU(),
        model_params: dict | None = None,
        seed: int = 0,
        baseline_growth_rate: float = 0.05,
        resistance_cost: float = 0.3,
        resistance_benefit: float = 1.0,
        network_updates_per_timepoint: int = 1,
        mutations_per_division: float = 0.01,
    ):
        self.is_cell = is_cell
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls
        self.activation_fxn = activation_fxn
        self.model_params = model_params

        first_agent = Agent(
            is_cell=is_cell,
            id=0,
            learning_rate=learning_rate,
            optimizer_cls=optimizer_cls,
            activation_fxn=activation_fxn,
            model_params=model_params,
            parent=None,
            n_cells=None if is_cell else 1,
        )
        self.agents = [first_agent]
        self.alive_ids: list[int] = [0]
        self.parent_ids: list[int | None] = [None]
        self.randomiser = npr.RandomState(seed)
        self.baseline_growth_rate = baseline_growth_rate
        self.resistance_cost = resistance_cost
        self.resistance_benefit = resistance_benefit
        self.network_updates_per_timepoint = network_updates_per_timepoint
        self.mutations_per_division = mutations_per_division

    @property
    def next_id(self):
        return len(self.parent_ids)

    @property
    def current_cell_count(self):
        if self.is_cell:
            return len(self.alive_ids)
        return sum(agent.n_cells for agent in self.agents)

    def run_simulation(
        self,
        detection_cell_count: int = 1000,
        n_timesteps_treatment: int = 20,
    ):
        os.makedirs("logs", exist_ok=True)
        with open("logs/cell_counts.csv", "w", encoding="utf-8") as f:
            f.write("timestep,cell_count,agent_count\n")
        with open("logs/cell_phenotypes.csv", "w", encoding="utf-8") as f:
            f.write("timestep,agent_id,n_cells,susceptible,resistant_0,resistant_1\n")
        if not self.is_cell:
            with open("logs/tree_structure.csv", "w", encoding="utf-8") as f:
                f.write("timestep,agent_id,parent_id,n_cells\n")
        timestep = 0
        while self.current_cell_count < detection_cell_count:
            self.advance_one_timestep(timestep, treatment=None)
            timestep += 1
        print(
            f"Detected {self.current_cell_count} cells at timestep {timestep}, running treatment 0"
        )
        for treatment in range(2):
            for i in range(n_timesteps_treatment):
                self.advance_one_timestep(
                    timestep + i + treatment * n_timesteps_treatment,
                    treatment=treatment,
                )
                if len(self.alive_ids) == 0:
                    print("All cells died; simulation complete")
                    return
                if len(self.alive_ids) > 4 * detection_cell_count:
                    print(
                        f"Detected {len(self.alive_ids)} cells at timestep {timestep + i}; "
                        f"patient has gained resistance and progressed"
                    )
                    return
            print(f"Ran treatment {treatment} for {n_timesteps_treatment} timesteps")

    def advance_one_timestep(self, timestep: int, treatment: int | None = None):
        growth_rates = []
        for alive_id in self.alive_ids:
            agent = self.agents[alive_id]
            assert agent is not None
            doses = get_doses_from_treatment(treatment)
            for _ in range(self.network_updates_per_timepoint):
                agent.update_phenotype(doses)
            relative_growth_rate = agent.calc_growth_rate(
                agent.phenotype,
                doses,
                self.resistance_cost,
                self.resistance_benefit,
            )
            growth_rate = self.baseline_growth_rate * relative_growth_rate
            growth_rates.append(growth_rate)
            if not self.is_cell:
                new_clones = agent.update_cell_count(
                    self.randomiser,
                    growth_rate,
                    self.mutations_per_division,
                    len(self.parent_ids),
                )
                self.agents.extend(new_clones)
                self.parent_ids.extend([alive_id] * len(new_clones))
                self.alive_ids.extend([clone.id for clone in new_clones])
                if agent.n_cells == 0:
                    self.alive_ids.remove(alive_id)
            else:
                if agent.dies(self.randomiser, growth_rate):
                    self.alive_ids.remove(alive_id)
                elif agent.divides(self.randomiser, growth_rate):
                    new_agent = agent.copy(new_id=len(self.agents))
                    mutate = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                    if mutate:
                        new_agent.mutate()

                    self.agents.append(new_agent)
                    self.parent_ids.append(alive_id)
                    self.alive_ids.append(agent_copy.id)
        if timestep % 10 == 0:
            print(f"growth rates: {np.mean(growth_rates)} ± {np.std(growth_rates)}")
        # log results
        # if timestep % 10 == 0:
        with open("logs/cell_counts.csv", "a", encoding="utf-8") as f:
            f.write(f"{timestep},{self.current_cell_count},{len(self.alive_ids)}\n")
        with open("logs/cell_phenotypes.csv", "a", encoding="utf-8") as f:
            for agent_id in self.alive_ids:
                agent = self.agents[agent_id]
                f.write(
                    f"{timestep},{agent.id},{agent.n_cells},{agent.phenotype[0]},{agent.phenotype[1]},"
                    f"{agent.phenotype[2]}\n"
                )
        if not self.is_cell:
            with open("logs/tree_structure.csv", "a", encoding="utf-8") as f:
                for agent_id in self.alive_ids:
                    agent = self.agents[agent_id]
                    assert (
                        agent.parent is None and self.parent_ids[agent_id] is None
                    ) or (
                        agent.parent is not None
                        and self.parent_ids[agent_id] == agent.parent.id
                    )
                    f.write(
                        f"{timestep},{agent.id},{agent.parent.id if agent.parent is not None else None},{agent.n_cells}\n"
                    )


def get_doses_from_treatment(treatment: int | None):
    return torch.from_numpy(
        np.array(
            [0.0 if (i != treatment or treatment is None) else 1.0 for i in range(2)],
            dtype=np.float32,
        )
    ).reshape(1, -1)


if __name__ == "__main__":
    phylogeny = Phylogeny(is_cell=True)
    phylogeny.run_simulation()
