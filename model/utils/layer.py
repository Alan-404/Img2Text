import torch
from torch import Tensor
import torch.nn as nn

from .attention import MutltiHeadAttention
from .ffn import PositionWiseFeedForward
from .residual_connection import ResidualConnection

from typing import Callable

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.masked_multi_head_attention = MutltiHeadAttention(heads=heads, d_model=d_model)
        self.mutli_head_attention = MutltiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = PositionWiseFeedForward(d_ff=d_ff, d_model=d_model, activation=activation)

        self.residual_connection_1 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_3 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)

    def forward(self, x: Tensor, encoder_output: Tensor, mask: Tensor):
        # sublayer 1
        q = k = v = x
        masked_attention_output = self.masked_multi_head_attention(q, k, v, mask)
        sub_layer_1 = self.residual_connection_1(masked_attention_output, x)

        # sublayer 2
        q = sub_layer_1
        k = v = encoder_output
        attention_output = self.mutli_head_attention(q, k, v, None)
        sub_layer_2 = self.residual_connection_2(attention_output, sub_layer_1)

        # sublayer 3
        ffn_inp = sub_layer_2
        ffn_output = self.ffn(ffn_inp)
        sub_layer_3 = self.residual_connection_3(ffn_output, sub_layer_3)

        return sub_layer_3

