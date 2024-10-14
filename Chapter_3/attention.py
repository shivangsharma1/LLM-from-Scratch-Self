import torch
import torch.nn as nn
torch.manual_seed(123)

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
    
#better to implement Q, K , V matrices using Linear layer because linear has good weight initialization technique as compared to torch.rand
class SelfAttention_lin(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.w_q = nn.Linear(d_in, d_out, bias=False)
        self.w_k = nn.Linear(d_in, d_out, bias=False)
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        query = self.w_q(x)
        key = self.w_k(x)
        val = self.w_v(x)

        #attention scores
        attn_score = query @ key.T
        key_dim = key.shape[-1]
        attn_weight = torch.softmax(attn_score/(key_dim**0.5), dim = -1)
        context_vec = attn_weight @ val
        return context_vec

#to product same weights as of linear
class SelfAttention_same_weight(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.w_q = nn.Parameter(attn_block.w_q.weight.T, requires_grad=False)
        self.w_k = nn.Parameter(attn_block.w_k.weight.T, requires_grad=False)
        self.w_v = nn.Parameter(attn_block.w_v.weight.T, requires_grad=False)

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


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.w_q = nn.Linear(d_in, d_out, bias=False)
        self.w_k = nn.Linear(d_in, d_out, bias=False)
        self.w_v = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        query = self.w_q(x)
        key = self.w_k(x)
        val = self.w_v(x)

        #attention scores
        attn_score = query @ key.T
        key_dim = key.shape[-1]
        attn_weight = torch.softmax(attn_score/(key_dim**0.5), dim = -1)
        

        #masking 
        context_len = attn_score.shape[0]
        mask = torch.tril(torch.ones(context_len, context_len))
        mask_attn_weight = attn_weight @ mask

        #normalizing 
        row_sum = mask_attn_weight.sum(dim=-1, keepdim=True)
        mask_attn_weight = mask_attn_weight / row_sum
        context_vec = mask_attn_weight @ val
        return context_vec


if __name__ == '__main__':
    d_in, d_out = 5, 5
    attn_block = SelfAttention_lin(d_in, d_out)
    att_block_same_weight = SelfAttention_same_weight(d_in, d_out)

    x_in = torch.rand(5, 5)
    print("first", attn_block.forward(x_in))
    print("second", att_block_same_weight.forward(x_in))


    #masked attention
    causal = CausalAttention(d_in, d_out)
    print("Causal", causal.forward(x_in))
