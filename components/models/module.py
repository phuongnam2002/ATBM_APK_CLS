import torch.nn as nn


class MLPLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.linear(x)

        return x
