import numpy as np
import torch
from torch import nn

from .Agent import Agent


class Phylogeny(object):
    def __init__(
        self,
        is_cell: bool = True,
        learning_rate: float = 1 * (10**-3),
        optimizer_cls: type = torch.optim.SGD,
        activation_fxn: nn.Module = nn.ReLU(),
        model_params: dict | None = None,
        seed: int = 0,
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

    def advance_one_timestep(self, treatment: int | None = None):
        for alive_cell_id in self.alive_ids:
            agent = self.agents[alive_cell_id]
            doses = get_doses_from_treatment(treatment)
            agent.update_phenotype(doses)
            if not self.is_cell:
                new_clones = agent.update_cell_count(self.randomiser)
                self.agents.extend(new_clones)
                self.parent_ids.extend([alive_cell_id] * len(new_clones))
            else:
                growth_rate = agent.calc_growth_rate(agent.phenotype, doses)
                if agent.dies(self.randomiser, growth_rate):
                    self.alive_ids.remove(alive_cell_id)
                elif agent.divides(self.randomiser, growth_rate):
                    agent_copy = agent.copy(new_id=len(self.agents))
                    self.agents.append(agent_copy)
                    self.parent_ids.append(alive_cell_id)


def get_doses_from_treatment(treatment: int | None):
    return torch.from_numpy(
        np.array(
            [0.0 if (i != treatment or treatment is None) else 1.0 for i in range(2)],
            dtype=np.float32,
        )
    ).reshape(1, -1)
