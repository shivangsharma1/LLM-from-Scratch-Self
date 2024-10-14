import sys
sys.path.append('/Users/Shivang/LLM from scratch/LLM-from-Scratch-Self/')

from regex import D
from Chapter_3.mha_efficient import MultiheadAttention
import torch
import torch.nn as nn
from gpt import LayerNorm, TansformersConfig
from gelu import Feedforward

global constants

class TransformersBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att  = MultiheadAttention(
            d_in=constants.emb_dim, 
            d_out = constants.emb_dim,
            context_len = constants.context_length,
            num_heads = constants.n_heads,
            dropout = constants.drop_rate,
            bias = constants.qkv_bias
        )
    
        self.ff = Feedforward(cfg)
        self.norm1 = LayerNorm(constants.emb_dim)
        self.norm2 = LayerNorm(constants.emb_dim)
        self.drop_shortcut = nn.Dropout(constants.drop_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x+shortcut

        return x




if __name__ == '__main__':
    
    constants = TansformersConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )

    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    block = TransformersBlock(constants)
    output = block(x)
    # print(output)

    print("input shape: ", x.shape)
    print("output shape: ", output.shape)