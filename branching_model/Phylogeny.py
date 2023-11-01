import numpy as np
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
        self.randomiser = np.random.RandomState(seed)
        self.baseline_growth_rate = baseline_growth_rate

    def run_simulation(
        self,
        detection_cell_count: int = 1000,
        n_timesteps_treatment: int = 20,
    ):
        os.makedirs("logs", exist_ok=True)
        with open("logs/cell_counts.log", "w", encoding="utf-8") as f:
            f.write("timestep,cell_count\n")
        timestep = 0
        while len(self.agents) < detection_cell_count:
            self.advance_one_timestep(timestep, treatment=None)
            timestep += 1
        print(
            f"Detected {len(self.agents)} cells at timestep {timestep}, running treatment 0"
        )
        for i in range(n_timesteps_treatment):
            self.advance_one_timestep(timestep + i, treatment=0)
        print(
            f"Ran treatment 0 for {n_timesteps_treatment} timesteps; running treatment 1"
        )
        for i in range(n_timesteps_treatment):
            self.advance_one_timestep(timestep + n_timesteps_treatment + i, treatment=1)
        print(
            f"Ran treatment 1 for {n_timesteps_treatment} timesteps; simulation complete"
        )

    def advance_one_timestep(self, timestep: int, treatment: int | None = None):
        growth_rates = []
        for alive_cell_id in self.alive_ids:
            agent = self.agents[alive_cell_id]
            doses = get_doses_from_treatment(treatment)
            agent.update_phenotype(doses)
            if not self.is_cell:
                new_clones = agent.update_cell_count(self.randomiser)
                self.agents.extend(new_clones)
                self.parent_ids.extend([alive_cell_id] * len(new_clones))
            else:
                relative_growth_rate = agent.calc_growth_rate(agent.phenotype, doses)
                growth_rate = self.baseline_growth_rate * relative_growth_rate
                growth_rates.append(growth_rate)
                if agent.dies(self.randomiser, growth_rate, self.baseline_growth_rate):
                    self.alive_ids.remove(alive_cell_id)
                elif agent.divides(
                    self.randomiser, growth_rate, self.baseline_growth_rate
                ):
                    agent_copy = agent.copy(new_id=len(self.agents))
                    self.agents.append(agent_copy)
                    self.parent_ids.append(alive_cell_id)
                    self.alive_ids.append(agent_copy.id)
        print(f"growth rates: {np.mean(growth_rates)} ± {np.std(growth_rates)}")
        # log results
        # if timestep % 10 == 0:
        with open("logs/cell_counts.log", "a", encoding="utf-8") as f:
            f.write(f"{timestep},{len(self.alive_ids)}\n")


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
