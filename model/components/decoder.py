import torch
from torch import Tensor
import torch.nn as nn

from model.utils.layer import DecoderLayer
from model.utils.position import PostionalEncoding
from model.utils.mask import generate_look_ahead_mask
from typing import Callable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):
    def __init__(self, token_size: int, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.positional_encoding = PostionalEncoding()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model, d_ff=d_ff, heads=heads, dropout_rate=dropout_rate, eps=eps, activation=activation) for _ in range(n)])
        self.linear = nn.Linear(in_features=d_model, out_features=token_size)
        self.to(device)

    def forward(self, x: Tensor ,encoder_otuput: Tensor):
        mask = generate_look_ahead_mask(x)
        x = self.embedding_layer(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x,encoder_otuput, mask)
        x = self.linear(x)
        return x