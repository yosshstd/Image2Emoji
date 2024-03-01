import torch
from torch import nn
import torch.nn.functional as F

class ProjectionHead2(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)

        x += projected

        return self.layer_norm(x)

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, projection_dim*2*2)
        self.geglu = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(projection_dim*2, projection_dim) 

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.geglu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)