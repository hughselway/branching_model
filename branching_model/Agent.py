from copy import deepcopy
import numpy as np
from torch import nn
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# Related to the network
N_LAYERS = 3
N_TREATMENTS = 2
N_IN = N_TREATMENTS
N_OUT = N_TREATMENTS + 1
N_HIDDEN_NEURONS = 2 * (N_OUT)
FUNNEL_S = 0.5

WT_IDX = 0

RESISTANCE_C = 0.1
RESISTANCE_B = 1


BP_MUTATION_RATE = 4*(10**-9)  # from Ben's paper, referenced by reviewer 1
BP_IN_EXOME = (45-18)*(10**6)  # 45Mb - 18Mb of potential synonymous mutations http://www.nature.com/nature/journal/v536/n7616/full/nature19057.html?foxtrotcallback=true
CLONE_MUTATION_RATE = 1 - stats.binom.pmf(k=0, n=BP_IN_EXOME, p=BP_MUTATION_RATE) # probability of at least 1 mutation
MUTATION_WEIGHT = 0.5 # Maximum amount of noise to add to new agent's network


def add_noise_to_weights(m, max_noise=0.25):
    """
    https://discuss.pytorch.org/t/is-there-any-way-to-add-noise-to-trained-weights/29829/5
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * max_noise)

class NN(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden_units: int,
        n_layers: int,
        n_cls: int,
        funnel_s: float,
        activation_fxn: nn.Module,
        loss_fxn,
        optimizer_cls: type,
        optimizer_init_kwargs: dict,
        device: str | None = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.n_cls = n_cls
        if self.n_cls == 2:
            out_features = 1
        else:
            out_features = self.n_cls

        self.activation_fxn = activation_fxn

        self.build_network(
            n_features=n_features,
            n_cls=out_features,
            n_hidden_units=n_hidden_units,
            n_layers=n_layers,
            funnel_s=funnel_s,
        )
        self.optimizer = self.get_optimizer(
            optimizer_cls=optimizer_cls, optimizer_init_kwargs=optimizer_init_kwargs
        )
        self.loss_fxn = loss_fxn

    def get_optimizer(self, optimizer_cls: type, optimizer_init_kwargs: dict):
        optimizer = optimizer_cls(self.parameters(), **optimizer_init_kwargs)
        return optimizer

    def build_network(
        self,
        n_features: int,
        n_cls: int,
        n_hidden_units: int,
        n_layers: int,
        funnel_s: float,
    ) -> None:
        self.layer_1 = nn.Linear(
            in_features=n_features, out_features=n_hidden_units
        )  # takes in n features (X), produces 5 features
        self.inner_layers = nn.ModuleList(
            [
                nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units)
                for i in range(n_layers)
            ]
        )

        # Create funnel layers
        if funnel_s > 0:
            if funnel_s < 1:
                n_funnel_layers = (
                    np.floor(np.log(n_cls / n_hidden_units) / np.log(funnel_s)).astype(
                        int
                    )
                    - 1
                )
                funnel_layer_widths = [
                    np.round(n_hidden_units * (funnel_s**x)).astype(int)
                    for x in range(1, n_funnel_layers + 1)
                ]
            elif funnel_s >= 1:
                n_funnel_layers = np.floor((n_hidden_units - n_cls) / funnel_s).astype(
                    int
                )
                funnel_layer_widths = [
                    n_hidden_units - np.round(funnel_s * x).astype(int)
                    for x in range(1, n_funnel_layers + 1)
                ]

            if n_funnel_layers <= 0:
                self.funnel_layers = None
                last_funnel_layer_w = n_hidden_units

            else:
                funnel_layer_widths = [n_hidden_units, *funnel_layer_widths]
                self.funnel_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            in_features=funnel_layer_widths[i],
                            out_features=funnel_layer_widths[i + 1],
                        )
                        for i in range(n_funnel_layers)
                    ]
                )
                last_funnel_layer_w = funnel_layer_widths[-1]

        else:
            self.funnel_layers = None
            last_funnel_layer_w = n_hidden_units
            n_funnel_layers = 0

        self.nn_funnel_layers = n_funnel_layers

        self.last_layer = nn.Linear(in_features=last_funnel_layer_w, out_features=n_cls)

    def forward(self, x: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        z = self.activation_fxn(self.layer_1(x))
        for layer in self.inner_layers:
            z = self.activation_fxn(layer(z))

        if self.funnel_layers is not None:
            for layer in self.funnel_layers:
                z = self.activation_fxn(layer(z))

        z = self.last_layer(z)
        if return_probs:
            z = self.logit2prob(z)

        return z

    def logit2prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_cls == 2:
            probs = torch.sigmoid(x.squeeze())
        else:
            dim = 1 if x.shape[0] > 1 else 0
            probs = torch.softmax(x.squeeze(), dim=dim)

        return probs


class Agent(object):
    def __init__(
        self,
        is_cell: bool,
        id: int,
        learning_rate: float = 1 * (10**-3),
        optimizer_cls: type = torch.optim.SGD,
        activation_fxn: nn.Module = nn.ReLU(),
        model_params: dict | None = None,
        parent: "Agent | None" = None,
        n_cells: int | None = None,
    ):
        """

        Parameters
        ----------
        optimizer_cls: torch.optim
            torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, etc...

        activation:
            nn.ReLU(), nn.LeakyReLU(), etc...

        """

        self.id = id
        self.parent = parent
        self.model = NN(
            n_features=N_TREATMENTS,
            n_cls=N_OUT,
            n_hidden_units=N_HIDDEN_NEURONS,
            n_layers=N_LAYERS,
            funnel_s=FUNNEL_S,
            activation_fxn=activation_fxn,
            loss_fxn=None,
            optimizer_cls=optimizer_cls,
            optimizer_init_kwargs={"lr": learning_rate},
        )
        if is_cell:
            assert n_cells is None
            self.status = "cell"
        else:
            self.status = "clone"
            self.n_cells = n_cells or 1

        if parent is not None:
            self.phenotype = parent.phenotype
        else:
            self.phenotype = self.model.forward(torch.zeros((1, N_TREATMENTS)))

    def update_phenotype(self, doses: torch.Tensor) -> None:
        self.model.train()
        new_pheno = self.model.forward(doses)

        loss = self.calc_loss(new_pheno, doses)  # fitness

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        self.model.eval()

        self.phenotype = new_pheno

    def calc_loss(self, pheno: torch.Tensor, doses: torch.Tensor) -> torch.Tensor:
        """
        minimize cost of treatment
        pheno: tensor
            Degree of resistance to each treatement. First value is for susceptible to all
        """

        susceptibility = (1 - pheno) / torch.sum(1 - pheno)  # 0-1
        treatment_effect = torch.sum(
            susceptibility[WT_IDX + 1 :] * doses
        )  # minimize this
        cost_of_resistance = torch.sum(
            pheno[WT_IDX + 1 :]
        )  # minimize this. Should select for susceptible when no drug
        fitness = RESISTANCE_B * treatment_effect + RESISTANCE_C * cost_of_resistance

        return fitness

    def mutate(self):
        """
        Mutation adds noise to inherited network and creates new optimizer
        """
        # self = cell
        self.model.apply(lambda m: add_noise_to_weights(m, MUTATION_WEIGHT))
        optimizer_init_kwargs = {"lr": self.model.optimizer.param_groups[0]["lr"]}
        optimizer_cls=type(self.model.optimizer)
        self.model.optimizer = self.model.get_optimizer(
            optimizer_cls=optimizer_cls,
            optimizer_init_kwargs=optimizer_init_kwargs)


    def calc_growth_rate(self, pheno: torch.Tensor, doses: torch.Tensor) -> float:
        return 1 - 2 * float(self.calc_loss(pheno, doses))

    def update_cell_count(self, randomiser: np.random.RandomState) -> list["Agent"]:
        """
        update number of cells according to current fitness
        returns list of new clones resulting from mutations (if any)
        """
        assert self.status == "clone"
        raise NotImplementedError

    def copy(self, new_id: int) -> "Agent":
        """
        returns a copy of the agent, with a new id and deepcopied model and optimizer
        """
        new_agent = Agent(
            is_cell=self.status == "cell",
            id=new_id,
            learning_rate=self.model.optimizer.param_groups[0]["lr"],
            optimizer_cls=type(self.model.optimizer),
            activation_fxn=self.model.activation_fxn,
            model_params=None,
            parent=self,
            n_cells=None if self.status == "cell" else self.n_cells,
        )
        new_agent.model.load_state_dict(deepcopy(self.model.state_dict()))
        # print("Not inheriting optimizer")
        new_agent.model.optimizer.load_state_dict(deepcopy(self.model.optimizer.state_dict()))

        return new_agent

    def dies(self, randomiser: np.random.RandomState, growth_rate: float) -> bool:
        """
        randomly decide if the cell dies
        """
        if growth_rate > 0:
            return False
        if randomiser.uniform() < growth_rate + 1:
            return True
        return False

    def divides(self, randomiser: np.random.RandomState, growth_rate: float) -> bool:
        """
        randomly decide if the cell divides
        """
        if growth_rate < 0:
            return False
        if randomiser.uniform() > growth_rate:
            return True
        return False
