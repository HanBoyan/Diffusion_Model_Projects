import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, dim_embed:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(dim_embed, 3 * dim_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(dim_embed, dim_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = dim_embed // n_heads

    def forward(self, x, causal_mask=False):
        #x:(batch_size, seq_len, dim_embed)
        input_shape = x.shape
        batch_size, seq_len, dim_embed = input_shape

        interm_shape = (batch_size, seq_len,  self.n_heads, self.d_head)

        #(batch_size,seq_len,dim_embed) -> (batch_size,seq_len,dim_embed*3) -> 3 tensors shaped of (batch_size,seq_len,dim_embed)
        q,k,v = self.in_proj(x).chunk(3, dim=-1)

        #(batch_size,seq_len,dim_embed) -> (batch_size,seq_len,n_heads,d_head) -> (batch_size,n_heads,seq_len,d_head)
        q = q.view(interm_shape).transpose(1,2)
        k = k.view(interm_shape).transpose(1,2)
        v = v.view(interm_shape).transpose(1,2)

        #(batch_size,n_heads,seq_len,d_head) @ (batch_size,n_heads,d_head,seq_len) -> (batch_size,n_heads,seq_len,seq_len)
        weight = q @ k.transpose(-1,-2)

        if causal_mask:
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)
        #(batch_size,n_heads,seq_len,seq_len) @ (batch_size,n_heads,seq_len,d_head) -> (batch_size,n_heads,seq_len,d_head)
        output = weight @ v
        #(batch_size,seq_len,n_heads,d_head) -> (batch_size,n_heads,seq_len,d_head)
        output = output.transpose(1,2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)
        return output