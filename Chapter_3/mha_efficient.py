import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_len, num_heads, dropout=0.1, bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_head = num_heads
        self.head_dim = d_out // num_heads

        self.q = nn.Linear(d_in, d_out, bias=bias)
        self.k = nn.Linear(d_in, d_out, bias=bias)
        self.v = nn.Linear(d_in, d_out, bias=bias)
        # linear layer to combine heads output
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # declaration for attention calculation, defining it as register_buffer so that it moves to \
        # model current device
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_len, context_len), diagonal=1)
        )  # mask

    def forward(self, x):
        b, token_len, d_in = x.shape
        key = self.k(x)
        query = self.q(x)
        val = self.v(x)

        # converting into MHA dim
        key = key.view(b, token_len, self.num_head, self.head_dim)
        query = query.view(b, token_len, self.num_head, self.head_dim)
        val = val.view(b, token_len, self.num_head, self.head_dim)

        # converting to channel dim
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        val = val.transpose(1, 2)

        # normal attention calculation
        attn_score = query @ key.transpose(2, 3)
        mask_bool = self.mask.bool()[:token_len, :token_len]

        # masking
        attn_score.masked_fill_(mask_bool, -torch.inf)

        atten_weight = torch.softmax(attn_score / (key.shape[-1] ** 0.5), dim=-1)
        atten_weight = self.dropout(atten_weight)

        context_vec = (atten_weight @ val).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, token_len, self.d_out)

        # passing the vector to output projection layer to combine the heads output, so this is also a trainable layer
        context_vec = self.out_proj(context_vec)

        return context_vec


if __name__ == "__main__":
    d_in, d_out = 3, 8

    x_in = torch.rand(5, 3)
    batch = torch.stack((x_in, x_in), dim=0)

    print("batch size", batch.shape)

    # masked attention
    mha = MultiheadAttention(d_in, d_out, batch.shape[1], num_heads=2)
    print("Causal", mha.forward(batch))
    print("Output shape", mha.forward(batch).shape)
