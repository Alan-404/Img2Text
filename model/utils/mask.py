import torch
from torch import Tensor
import numpy as np

def generate_padding_mask(x: Tensor):
    return torch.Tensor(x == 0).type(torch.int64)[:, np.newaxis, np.newaxis, :]


def __generate_trig(length: int):
    return torch.triu(torch.ones((length, length)), diagonal=-1)

def generate_look_ahead_mask(x: Tensor):
    padding_mask = generate_look_ahead_mask(x)
    trig_matrix = __generate_trig(x.size(1))

    look_ahead_mask = torch.maximum(trig_matrix, padding_mask)

    return look_ahead_mask