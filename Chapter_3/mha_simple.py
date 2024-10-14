from causal_drop_attention import CausalAttention
import torch 
import torch.nn as nn

class MultiHeadAttentionWrapperSeq(nn.Module):
    def __init__(self, d_in, d_out, context_len, num_heads ,dropout = 0.5, bias = False):
        super().__init__()
        self.head = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_len) for _ in range(num_heads)]
        )

    def forward(self, x):
        #Sequential implememtation
        return torch.cat([head(x) for head in self.head], dim = -1)
    


if __name__ == "__main__":
    d_in, d_out = 3, 2

    x_in = torch.rand(5, 3)
    batch = torch.stack((x_in, x_in), dim=0)

    print("batch size", batch.shape)

    # masked attention
    mha = MultiHeadAttentionWrapperSeq(d_in, d_out, batch.shape[1], num_heads=2)
    print("Causal", mha.forward(batch))
    print("Output shape", mha.forward(batch).shape)
