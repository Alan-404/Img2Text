import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
    dk = torch.tensor(k.size(-1), dtype=torch.float32)

    attention_scores = torch.matmul(q, k.transpose(-1, -2))
    attention_scores = attention_scores/(torch.sqrt(dk))

    if mask is not None:
        attention_scores = attention_scores + mask*(-1e15)

    attention_weights = F.softmax(attention_scores, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output

class MutltiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        self.head_samples = self.d_model // self.heads

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

        self.to(device)

    
    
    def split(self, x: Tensor):
        batch_size = x.size(0)
        length = x.size(1)

        x = x.reshape((batch_size, length, self.heads, self.head_samples))
        x = x.permute((0, 2, 1, 3))

        return x
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        batch_size = q.size(0)
        length = q.size(1)

        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        q_heads = self.split(qw)
        k_heads = self.split(kw)
        v_heads = self.split(vw)

        attention_output = scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)
        
        attention_output = attention_output.permute((0, 2, 1, 3))
        attention_output = attention_output.reshape((batch_size, length, self.d_model))

        output = self.linear_output(attention_output)

        return output





