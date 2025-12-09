import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-1], 1)  # 输出 logit（未过 sigmoid）

    def forward(self, x):
        h = self.mlp(x)
        logit = self.out(h).squeeze(-1)  # [batch]
        return logit
