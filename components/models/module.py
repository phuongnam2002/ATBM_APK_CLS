import torch.nn as nn
from transformers import PretrainedConfig


class MLPLayer(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, x):
        x = self.linear(x)

        return x
