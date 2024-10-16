import sys
from torch import manual_seed

sys.path.append("/Users/Shivang/LLM from scratch/LLM-from-Scratch-Self/")

from Chapter_4.gpt import GPTModel, TansformersConfig, generate_text
import tiktoken
import torch
import torch.nn as nn

torch.manual_seed(123)


# text to token_ids
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded = torch.tensor(encoded).unsqueeze(0)  # adding batch dimension
    return encoded


# token to ids
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())  # removing the batch dimension


if __name__ == "__main__":

    contants = TansformersConfig(
        vocab_size=50257,
        context_length=256,
        emb_dim=768,
        n_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(contants)
    model.eval()
    token_ids = generate_text(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10, 
        context_size=contants.context_length
    )

    print("output text: \n ", token_ids_to_text(token_ids, tokenizer))