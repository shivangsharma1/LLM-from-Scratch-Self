import tokenize
from numpy import arange
from sympy import false
import tiktoken
import torch

from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def pos_embedding(token_embedding, context_length=4, vocab_size=50257, output_dim=256):
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    pos_embedding = token_embedding_layer(
        torch.arange(context_length)
    )  # absolute positional embedding
    token_embedding_v1 = token_embedding_layer(token_embedding)

    # adding both embedding
    input_embedding = token_embedding_v1 + pos_embedding
    print("input_embeddding shape", input_embedding.shape)


if __name__ == "__main__":
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        data = f.read()

    dataloader = create_dataloader_v1(
        data, batch_size=8, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader)
    first_batch = next(data_iter)

    X, y = first_batch
    print(f"X: {X}")
    print(f"y: {y}")

    # input token embeddings
    pos_embedding(X)
