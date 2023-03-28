import torch
from torch import Tensor
import torch.nn as nn

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        self.norm_layer = nn.LayerNorm(normalized_shape=d_model, eps=eps)

    def forward(self, x:Tensor, pre_x: Tensor):
        x = self.dropout_layer(x)
        x = x + pre_x
        x = self.norm_layer(x)

        return x
