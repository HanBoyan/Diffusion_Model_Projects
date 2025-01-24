import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size,embedding_dim,seq_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len,embedding_dim))

    def forward(self, tokens: torch.LongTensor)->torch.Tensor:
        #(batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        tokens_embedding = self.token_embedding(tokens)
        tokens_embedding += self.position_embedding
        return tokens_embedding


class CLIPLayer(nn.Module):
    def __init__(self, n_heads, embedding_dim):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(embedding_dim)
        self.attn = SelfAttention(n_heads,embedding_dim)
        self.layernorm_2 = nn.LayerNorm(embedding_dim)
        self.linear_1 = nn.Linear(embedding_dim,embedding_dim*4)
        self.linear_2 = nn.Linear(embedding_dim*4,embedding_dim)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        #(batch_size, seq_len, embedding_dim)
        residue = x

        #self-attention
        x = self.layernorm_1(x)
        x = self.attn(x,causal_mask=True)
        x = x + residue

        #feedforward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x*torch.sigmoid(1.702*x)#quickGELU
        x = self.linear_2(x)
        x = x + residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(49408,768,77)
        self.layers = nn.ModuleList([
            CLIPLayer(12,768) for _ in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor)->torch.Tensor:
        tokens = tokens.type(torch.long)

        #(batch_size, seq_len) -> (batch_size, seq_len, 768)
        states = self.embedding(tokens)
        for layer in self.layers:
            states = layer(states)
        #(batch_size, seq_len, 768)
        output = self.layernorm(states)

        return output