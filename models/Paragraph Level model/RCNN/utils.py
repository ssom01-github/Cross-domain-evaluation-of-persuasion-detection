
#code for paragraph level data

import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")

def yield_tokens(df):
    # use the correct column name ("Text")
    for text in df["text"]:
        yield tokenizer(str(text))  # ensure text is string

def build_vocab(df):
    all_tokens = yield_tokens(df)
    vocab = build_vocab_from_iterator(all_tokens, specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab

class TextDataset(Dataset):
    def __init__(self, df, vocab, max_len=500):
        self.vocab = vocab
        self.max_len = max_len
        self.data = df.dropna(subset=["text"])

        # text column → tokens → vocab indices
        self.texts = self.data["text"].apply(lambda x: vocab(tokenizer(str(x))))
        self.labels = self.data["label"].astype(int)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts.iloc[idx]
        tokens = tokens[:self.max_len]  # truncate if too long
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)

