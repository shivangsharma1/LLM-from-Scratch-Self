import re

class SimpleTokenizerV2:
    def __init__(self, data):
        data = re.split(r'([,.:;?_!"()\']|--|\s)', data)
        all_tokens = sorted(list(set(data)))
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
        preprocessed = [item.strip() for item in all_tokens if item.strip()]
        vocab = {token: integer for integer, token in enumerate(preprocessed)}

        self.str_to_int = vocab
        self.int_to_str = {i:s for s , i in self.str_to_int.items()}

    
    def encode(self, text):
        text_preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in text_preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return text
    

if __name__ == '__main__':

    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        data = file.read()

    text = "Hello, world. Is this-- a test?"
    tokenizer = SimpleTokenizerV2(data)
    print(tokenizer.encode(text))
    print(tokenizer.decode(tokenizer.encode(text)))

