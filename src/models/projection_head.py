import torch
from torch import nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()

        self.linear1 = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(projection_dim, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        residual = self.linear1(x)
        x = self.gelu(residual)
        x = self.linear2(x)
        x = self.dropout(x)

        x += residual

        return self.layer_norm(x)