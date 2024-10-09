import torch
import torch.nn as nn


#implementing attention class

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.w_q = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_k = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        self.w_v = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    def forward(self, x):
        query = x @ self.w_q
        key = x @ self.w_k
        val = x @ self.w_v

        #attention scores
        attn_score = query @ key.T
        key_dim = key.shape[-1]
        attn_weight = torch.softmax(attn_score/(key_dim**0.5), dim = -1)
        context_vec = attn_weight @ val
        return context_vec
    

if __name__ == '__main__':
    d_in, d_out = 5, 50
    attn_block = SelfAttention(d_in, d_out)

    x_in = torch.rand(5, 5)
    print(attn_block.forward(x_in))
