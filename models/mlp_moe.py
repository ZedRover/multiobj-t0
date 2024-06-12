from typing import *
import torch as th
from torch import nn, Tensor


# Expert model
class Expert(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        dropout: float = 0.2,
    ):
        super(Expert, self).__init__()
        dnn_layers = []
        hidden_sizes = [input_size] + hidden_sizes

        for i in range(len(hidden_sizes) - 1):
            dnn_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if i != len(hidden_sizes) - 2:
                dnn_layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
                dnn_layers.append(nn.LeakyReLU())
                dnn_layers.append(nn.Dropout(dropout))

        self.dnn_layers = nn.Sequential(*dnn_layers)

    def forward(self, x):
        return self.dnn_layers(x)


# Tower model
class Tower(nn.Module):
    def __init__(
        self,
        input_size: int,
        tower_sizes: List[int],
        dropout: float = 0.2,
    ):
        super(Tower, self).__init__()
        layers = []
        sizes = (
            [input_size] + tower_sizes + [1]
        )  # Output size is 1 for final regression output

        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# MOE model
class MOE(nn.Module):
    def __init__(
        self,
        input_size: int,
        shared_size: int,
        num_expert: int,
        expert_sizes: List[int],
        tower_sizes: List[int],
        num_tasks: int,
        dropout: float = 0.2,
    ):
        super(MOE, self).__init__()

        self.shared_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, shared_size),
            nn.BatchNorm1d(shared_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

        self.experts = nn.ModuleList(
            [Expert(shared_size, expert_sizes, dropout) for i in range(num_expert)]
        )

        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(shared_size, num_expert), nn.Softmax(dim=1))
                for _ in range(num_tasks)
            ]
        )

        self.towers = nn.ModuleList(
            [Tower(expert_sizes[-1], tower_sizes, dropout) for _ in range(num_tasks)]
        )

    def forward(self, x):
        shared_output = self.shared_layer(x)

        expert_outputs = [expert(shared_output) for expert in self.experts]
        expert_outputs = th.stack(
            expert_outputs, dim=1
        )  # Shape: (batch_size, num_expert, expert_output_size)

        outputs = []
        for gate, tower in zip(self.gates, self.towers):
            gate_output = gate(shared_output)  # Shape: (batch_size, num_expert)
            gate_output = gate_output.unsqueeze(2)  # Shape: (batch_size, num_expert, 1)
            weighted_expert_output = (expert_outputs * gate_output).sum(
                dim=1
            )  # Shape: (batch_size, expert_output_size)
            output = tower(weighted_expert_output)  # Shape: (batch_size, 1)
            outputs.append(output)

        return th.stack(outputs, dim=1)


# Example usage
if __name__ == "__main__":
    input_size = 101
    shared_size = 20
    num_expert = 3
    expert_sizes = [30, 40]
    tower_sizes = [50, 30]
    num_tasks = 2  # Number of tasks

    model = MOE(
        input_size, shared_size, num_expert, expert_sizes, tower_sizes, num_tasks
    )
    print(model)
    sample_input = th.randn(512, input_size)
    outputs = model(sample_input)
    # for i, output in enumerate(outputs):
    # print(f"Output {i+1}: {output}")
