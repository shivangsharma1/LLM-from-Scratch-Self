import sys
sys.path.append('/Users/Shivang/LLM from scratch/LLM-from-Scratch-Self/')

from sympy import false
import torch
import torch.nn as nn

# import cfg
import tiktoken
from dataclasses import dataclass

from Chapter_3.mha_efficient import MultiheadAttention


@dataclass
class TansformersConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

class Feedforward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(seff, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


class TransformersBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.att  = MultiheadAttention(
            d_in=cfg.emb_dim, 
            d_out = cfg.emb_dim,
            context_len = cfg.context_length,
            num_heads = cfg.n_heads,
            dropout = cfg.drop_rate,
            bias = cfg.qkv_bias
        )
    
        self.ff = Feedforward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

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

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_embedding = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.drop_rate)

        self.transformer_block = nn.Sequential(
            *[TransformersBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(
            cfg.emb_dim, cfg.vocab_size, bias=False
        )  # mapping it back to to the vocab size of the model to get the \
        # probs per word

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        token_embedding = self.token_embedding(in_idx)
        pos_embeddding = self.pos_embedding(torch.arange(seq_len, device=in_idx.device))

        # combining the embedding
        x = token_embedding + pos_embeddding
        x = self.drop(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


if __name__ == "__main__":

    contants = TansformersConfig(
        vocab_size=50257,
        context_length=1024,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))

    batch = torch.stack(batch, dim=0)
    model = GPTModel(contants)
    logits = model(batch)
    print("Output", logits)
    print("Logits shape", logits.shape)
