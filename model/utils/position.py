import torch
from torch import Tensor
import torch.nn as nn

class PostionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def __encode_length(self, length: int):
        pos = torch.arange(length)
        return pos.unsqueeze(-1).type(torch.float32)
    
    def __encode_embedding(self, embedding_dim: int):
        angles = torch.arange(embedding_dim)
        angles[1::2] = angles[0::2]
        angles = 1/(torch.pow(10000, angles/embedding_dim))

        return angles.unsqueeze(0)
    
    def forward(self, x: Tensor):
        pos = self.__encode_length(x.size(1))
        angles = self.__encode_embedding(x.size(2))

        pos_angles = torch.matmul(pos, angles)
        pos_angles[0::2] = torch.sin(pos_angles[0::2])
        pos_angles[1::2] = torch.cos(pos_angles[1::2])

        x = x + pos_angles.unsqueeze(0)

        return x