import numpy as np
import torch
from torch import nn

from .Agent import Agent
from .io import Recorder


RECORD_FREQ = 1 # Record interval
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

        self.time = 0
        self.live_agent_recorder = Recorder()

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
        # self.dead_ids: list[int | None] = None (wee need to keep track of the dead cells "to mirror what we get from ctDNA")
        self.parent_ids: list[int | None] = [None]
        self.randomiser = np.random.RandomState(seed)

    def advance_one_timestep(self, treatment: int | None = None):

        if self.time % RECORD_FREQ == 0:
            self.live_agent_recorder.record_time_pt(self.agents)

        self.time += 1
        for alive_cell_id in self.alive_ids:
            agent = self.agents[alive_cell_id]
            doses = get_doses_from_treatment(treatment)
            agent.update_phenotype(doses)
            if not self.is_cell:
                # Clones
                new_clones = agent.update_cell_count(self.randomiser)
                n_new_clones = agent.update_cell_count(doses)
                for i in range(n_new_clones):
                    new_agent = agent.copy(new_id=len(self.agents))
                    new_agent.mutate()
                    self.agents.append(new_agent)
                    self.parent_ids.append(alive_cell_id)                
            else:
                growth_rate = agent.calc_growth_rate(agent.phenotype, doses)
                self.dead_cell_ids = []
                if agent.dies(self.randomiser, growth_rate):
                    self.alive_ids.remove(alive_cell_id)
                    self.dead_cell_ids.append(alive_cell_id)
                elif agent.divides(self.randomiser, growth_rate):
                    new_agent = agent.copy(new_id=len(self.agents))
                    mutate = np.random.binomial(1, Agent.CLONE_MUTATION_RATE) == 1
                    if mutate:
                        new_agent.mutate()

                    self.agents.append(new_agent)
                    self.parent_ids.append(alive_cell_id)
        # for dead_cell_id in self.dead_cell_ids:
        #     return dead_cell_id


def get_doses_from_treatment(treatment: int | None):
    return torch.from_numpy(
        np.array(
            [0.0 if (i != treatment or treatment is None) else 1.0 for i in range(2)],
            dtype=np.float32,
        )
    ).reshape(1, -1)
