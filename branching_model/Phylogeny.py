import numpy as np
import numpy.random as npr
import os
import torch
from torch import nn

from branching_model.Agent import Agent
from branching_model.Recorder import Recorder

RECORD_FREQ = 1  # Record interval

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
        number_of_treatments: int = 2,
        turnover: float = 0.0,
    ):

        self.is_cell = is_cell
        self.learning_rate = learning_rate
        self.optimizer_cls = optimizer_cls
        self.activation_fxn = activation_fxn
        self.model_params = model_params

        self.time = 0
        self.max_id = 0
        self.max_clone_id = 0
        # self.max_cell_id = 0
        self.live_agent_recorder = Recorder()
        self.dead_agent_recorder = Recorder()

        first_agent = Agent(
            is_cell=is_cell,
            id=self.max_id,
            clone_id=self.max_clone_id,
            cell_id=self.max_cell_id,
            learning_rate=learning_rate,
            optimizer_cls=optimizer_cls,
            activation_fxn=activation_fxn,
            model_params=model_params,
            parent=None,
            n_cells=None if is_cell else 1,
        )
        self.agents = [first_agent]
        self.dead_agents = []
        self.alive_ids: list[int] = [0]
        # self.dead_ids: list[int | None] = None (wee need to keep track of the dead cells "to mirror what we get from ctDNA")
        self.parent_ids: list[int | None] = [None]
        self.randomiser = npr.RandomState(seed)
        self.baseline_growth_rate = baseline_growth_rate
        self.resistance_cost = resistance_cost
        self.resistance_benefit = resistance_benefit
        self.network_updates_per_timepoint = network_updates_per_timepoint
        self.mutations_per_division = mutations_per_division
        self.number_of_treatments = number_of_treatments
        self.turnover = turnover

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
        detection_treatment_delay: int = 0,
        # n_timesteps_treatment: int = 20,
        measure_tumour_every_n_timesteps: int = 7,
        tumour_measurement_delay: int = 0,
        max_cycles: int = 1,
    ):
        os.makedirs("logs", exist_ok=True)
        with open("logs/cell_counts.csv", "w", encoding="utf-8") as f:
            f.write("timestep,cell_count,agent_count\n")
        with open("logs/cell_phenotypes.csv", "w", encoding="utf-8") as f:
            f.write(
                "timestep,agent_id,n_cells,susceptible,"
                + ",".join([f"R{i}" for i in range(1, self.number_of_treatments + 1)])
                + "\n"
            )
        with open("logs/cycle_times.csv", "w", encoding="utf-8") as f:
            f.write("cycle,treatment,timesteps_to_detection\n")
        with open("logs/birth_death_counts.csv", "w", encoding="utf-8") as f:
            f.write("timestep,birth_count,death_count\n")
        if not self.is_cell:
            with open("logs/tree_structure.csv", "w", encoding="utf-8") as f:
                f.write("timestep,agent_id,parent_id,n_cells\n")
        while self.current_cell_count < detection_cell_count:
            self.advance_one_timestep(treatment=None)
        print(
            f"Detected {self.current_cell_count} cells at timestep {self.time}, running treatment 0"
            + (
                f" after {detection_treatment_delay} timesteps"
                if detection_treatment_delay > 0
                else ""
            )
        )
        for i in range(detection_treatment_delay):
            self.advance_one_timestep(treatment=None)

        cycle_count = 0
        while cycle_count < max_cycles:
            cycle_count += 1
            print(f"Cycle {cycle_count}")
            for treatment in range(self.number_of_treatments):
                # for i in range(n_timesteps_treatment):
                timestep_this_treatment = 0
                last_tumour_measurement: int | None = None
                while True:
                    timestep_this_treatment += 1
                    self.advance_one_timestep(treatment=treatment)
                    if len(self.alive_ids) == 0:
                        print("All cells died; simulation complete")
                        return
                    if len(self.alive_ids) > 4 * detection_cell_count:
                        print(
                            f"Detected {len(self.alive_ids)} cells at timestep {self.time + i}; "
                            f"patient has gained resistance and progressed"
                        )
                        return
                    if timestep_this_treatment % measure_tumour_every_n_timesteps == 0:
                        if (
                            last_tumour_measurement is not None
                            and self.current_cell_count > last_tumour_measurement
                        ):
                            # Tumour has grown; move to next treatment
                            break
                        last_tumour_measurement = self.current_cell_count
                for i in range(tumour_measurement_delay):
                    self.advance_one_timestep(treatment=treatment)
                print(
                    f"Ran treatment {treatment} for {timestep_this_treatment} timesteps, "
                    f"then waited {tumour_measurement_delay} timesteps; "
                    f"tumour size {self.current_cell_count}"
                )
                with open("logs/cycle_times.csv", "a", encoding="utf-8") as f:
                    f.write(f"{cycle_count},{treatment},{timestep_this_treatment}\n")
            if len(self.alive_ids) > 4 * detection_cell_count:
                print(
                    f"Detected {len(self.alive_ids)} cells at timestep {self.time}; "
                    f"patient has gained resistance and progressed"
                )
                break

        print("Simulation complete")
        self.live_agent_recorder.write_csv(dst_dir="logs", prefix="live")
        self.dead_agent_recorder.write_csv(dst_dir="logs", prefix="dead")

    def advance_one_timestep(self, treatment: int | None):
        doses = get_doses_from_treatment(treatment, self.number_of_treatments)
        if self.time % RECORD_FREQ == 0:
            self.live_agent_recorder.record_time_pt(self.agents, self.time, doses)

        self.time += 1

        self.dead_agents = []
        growth_rates = []
        death_count = 0
        division_count = 0
        for alive_id in self.alive_ids:
            agent = self.agents[alive_id]
            assert agent is not None
            # doses = get_doses_from_treatment(treatment, self.number_of_treatments)
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
                (
                    new_clones,
                    internal_division_count,
                    internal_death_count,
                ) = agent.update_cell_count(
                    self.randomiser,
                    growth_rate,
                    self.mutations_per_division,
                    len(self.parent_ids),
                    self.turnover,
                )
                division_count += internal_division_count
                division_count += len(new_clones)
                death_count += internal_death_count
                # self.agents.extend(new_clones)
                # self.parent_ids.extend([alive_id] * len(new_clones))
                # self.alive_ids.extend([clone.id for clone in new_clones])
                if agent.n_cells == 0:
                    self.alive_ids.remove(alive_id)
                    self.dead_agents.append(agent)

                for i in range(new_clones):
                    self.max_id += 1
                    self.max_clone_id += 1
                    new_agent = agent.copy(id=self.max_id, new_clone_id=self.max_clone_id)
                    new_agent.mutate()
                    new_agent.n_cells = 1

                    self.agents.append(new_agent)
                    self.parent_ids.append(alive_id)
                    self.alive_ids.append(new_agent.id)

            else:
                if agent.dies(self.randomiser, growth_rate, self.turnover):
                    death_count += 1
                    self.alive_ids.remove(alive_id)
                    self.dead_agents.append(agent)
                elif agent.divides(self.randomiser, growth_rate, self.turnover):
                    mutate = self.randomiser.random() < self.mutations_per_division
                    division_count += 1
                    self.max_cell_id += 1
                    self.max_id += 1

                    if mutate:
                        self.max_clone_id += 1
                        new_clone_id = self.max_clone_id
                    else:
                        # Same clone as parent since no mutation
                        new_clone_id = agent.clone_id

                    new_agent = agent.copy(id=self.max_id, new_clone_id=new_clone_id)
                    if mutate:
                        new_agent.mutate()

                    self.agents.append(new_agent)
                    self.parent_ids.append(alive_id)
                    self.alive_ids.append(new_agent.id)

        if self.time % RECORD_FREQ == 0 and len(self.dead_agents) > 0:
            self.dead_agent_recorder.record_time_pt(self.dead_agents, self.time, doses)

        if self.time % 10 == 0:
            print(f"growth rates: {np.mean(growth_rates)} ± {np.std(growth_rates)}")
        # log results
        # if timestep % 10 == 0:
        with open("logs/cell_counts.csv", "a", encoding="utf-8") as f:
            f.write(f"{self.time},{self.current_cell_count},{len(self.alive_ids)}\n")
        with open("logs/cell_phenotypes.csv", "a", encoding="utf-8") as f:
            for agent_id in self.alive_ids:
                agent = self.agents[agent_id]
                f.write(
                    f"{self.time},{agent.id},{agent.n_cells},"
                    + ",".join([str(x) for x in agent.phenotype.detach().numpy()])
                    + "\n"
                )
        with open("logs/birth_death_counts.csv", "a", encoding="utf-8") as f:
            f.write(f"{self.time},{division_count},{death_count}\n")
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
                        f"{self.time},{agent.id},{agent.parent.id if agent.parent is not None else None},{agent.n_cells}\n"
                    )


def get_doses_from_treatment(treatment: int | None, number_of_treatments: int):
    return torch.from_numpy(
        np.array(
            [
                0.0 if (i != treatment or treatment is None) else 1.0
                for i in range(number_of_treatments)
            ],
            dtype=np.float32,
        )
    ).reshape(1, -1)


if __name__ == "__main__":
    phylogeny = Phylogeny(is_cell=True)
    phylogeny.run_simulation()
