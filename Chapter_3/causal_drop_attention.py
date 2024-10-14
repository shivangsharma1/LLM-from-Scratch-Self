from multiprocessing import context
import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, dropout=0.5, bias=False):
        super().__init__()
        self.d_out = d_out
        self.q = nn.Linear(d_in, d_out, bias=bias)
        self.k = nn.Linear(d_in, d_out, bias=bias)
        self.v = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )

    def forward(self, x):
        b, num_token, d_in = x.shape
        query = self.q(x)
        key = self.k(x)
        val = self.v(x)

        # attention score
        attn_score = query @ torch.transpose(key, 1, 2)
        attn_score.masked_fill_(self.mask.bool()[:num_token, :num_token], -torch.inf)

        # attn_weights
        attn_weights = torch.softmax(attn_score / (key.shape[-1] ** 0.5), dim=-1)

        # dropout
        attn_weights = self.dropout(attn_weights)

        # context vec
        context_vec = attn_weights @ val

        return context_vec

if __name__ == "__main__":
    d_in, d_out = 5, 5

    x_in = torch.rand(5, 5)
    batch = torch.stack((x_in, x_in), dim=0)

    print("batch size", batch.shape)

    # masked attention
    causal = CausalAttention(d_in, d_out, batch.shape[1])
    print("Causal", causal.forward(batch))
