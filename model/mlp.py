import torch
from torch import nn
from torch.nn import functional as F


class MLP2(nn.Module):
    def __init__(self):
        super(MLP2, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([.6]))

    def forward(self, I, D):
        x = self.alpha * I + (1 - self.alpha) * D
        return x


class MLP(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=80, to_score=True):
        super(MLP, self).__init__()
        self.to_score = to_score
        # self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        if self.to_score:
            self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.to_score:
            x = self.fc2(x)
        return x
