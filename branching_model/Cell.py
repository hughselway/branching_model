import numpy as np
from torch import nn
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Related to the network
N_LAYERS = 3
N_TREATMENTS = 2
N_IN = N_TREATMENTS
N_OUT = N_TREATMENTS + 1
N_HIDDEN_NEURONS = 2*(N_OUT)
FUNNEL_S = 0.5

WT_IDX = 0

RESISTANCE_C = 0.1
RESISTANCE_B = 1


class NN(nn.Module):
    def __init__(self, n_features, n_hidden_units, n_layers, n_cls, funnel_s, activation_fxn, loss_fxn, optimizer_cls, optimizer_init_kwargs, device=None):
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

        self.build_network(n_features=n_features, n_cls=out_features, n_hidden_units=n_hidden_units, n_layers=n_layers, funnel_s=funnel_s)
        self.optimizer = self.get_optimizer(optimizer_cls=optimizer_cls,
                                            optimizer_init_kwargs=optimizer_init_kwargs)
        self.loss_fxn = loss_fxn

    def get_optimizer(self, optimizer_cls, optimizer_init_kwargs):
        optimizer = optimizer_cls(self.parameters(), **optimizer_init_kwargs)
        return optimizer

    def build_network(self, n_features, n_cls, n_hidden_units, n_layers, funnel_s):

        self.layer_1 = nn.Linear(in_features=n_features, out_features=n_hidden_units) # takes in n features (X), produces 5 features
        self.inner_layers = nn.ModuleList([nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units) for i in range(n_layers)])

        # Create funnel layers
        if funnel_s > 0:
            if funnel_s < 1:
                n_funnel_layers = np.floor(np.log(n_cls/n_hidden_units)/np.log(funnel_s)).astype(int) - 1
                funnel_layer_widths = [np.round(n_hidden_units*(funnel_s**x)).astype(int) for x in range(1, n_funnel_layers+1)]
            elif funnel_s >= 1:
                n_funnel_layers = np.floor((n_hidden_units-n_cls)/funnel_s).astype(int)
                funnel_layer_widths = [n_hidden_units-np.round(funnel_s*x).astype(int) for x in range(1, n_funnel_layers+1)]

            if n_funnel_layers <= 0:
                self.funnel_layers = None
                last_funnel_layer_w = n_hidden_units

            else:
                funnel_layer_widths = [n_hidden_units, *funnel_layer_widths]
                self.funnel_layers = nn.ModuleList([nn.Linear(in_features=funnel_layer_widths[i], out_features=funnel_layer_widths[i+1]) for i in range(n_funnel_layers)])
                last_funnel_layer_w = funnel_layer_widths[-1]

        else:
            self.funnel_layers = None
            last_funnel_layer_w = n_hidden_units
            n_funnel_layers = 0

        self.nn_funnel_layers = n_funnel_layers

        self.last_layer = nn.Linear(in_features=last_funnel_layer_w, out_features=n_cls)

    def forward(self, x, return_probs=True):
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

    def logit2prob(self, x):
        if self.n_cls == 2:
            probs = torch.sigmoid(x.squeeze())
        else:
            dim = 1 if x.shape[0] > 1 else 0
            probs = torch.softmax(x.squeeze(), dim=dim)

        return probs


class Cell(object):

    def __init__(self, id, learning_rate=1*(10**-3), optimizer_cls=torch.optim.SGD, activation_fxn=nn.ReLU(), model_params=None, parent=None):
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
        self.model = NN(n_features=N_TREATMENTS, n_cls=N_OUT,
                        n_hidden_units=N_HIDDEN_NEURONS, n_layers=N_LAYERS,
                        funnel_s=FUNNEL_S,
                        activation_fxn=activation_fxn,
                        loss_fxn=None,
                        optimizer_cls=optimizer_cls,
                        optimizer_init_kwargs={"lr": learning_rate}
                        )
        if parent is not None:
            self.phenotype = parent.phenotype
        else:
            self.phenotype = self.model.forward(torch.zeros((1, N_TREATMENTS)))

    def update_phenotype(self, doses):
        self.model.train()
        new_pheno = self.model.forward(doses)

        loss = self.calc_loss(new_pheno, doses) # fitness

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        self.model.eval()

        self.phenotype = new_pheno

    def calc_loss(self, pheno, doses):
        """
        minimize cost of treatment
        pheno: tensor
            Degree of resistance to each treatement. First value is for susceptible to all
        """

        susceptibility = (1-pheno)/torch.sum(1-pheno) #0-1
        treatment_effect = torch.sum(susceptibility[WT_IDX+1:]*doses) # minimize this
        cost_of_resistance = sum(pheno[WT_IDX+1:]) # minimize this. Should select for susceptible when no drug
        fitness = RESISTANCE_B*treatment_effect + RESISTANCE_C*cost_of_resistance

        return fitness

