#code for sentence level data 
# import pandas as pd
# import torch
# from torchtext.data.utils import get_tokenizer
# from torch.utils.data import Dataset
# import torchtext
# from torchtext.vocab import build_vocab_from_iterator


# tokenizer = get_tokenizer("basic_english")

# def yield_tokens(df):
#     for text in df["text"]:
#         yield tokenizer(text)

# def build_vocab(df):
#     all_tokens = yield_tokens(df)
#     vocab = build_vocab_from_iterator(all_tokens, specials=["<pad>", "<unk>"])
#     vocab.set_default_index(vocab["<unk>"])
#     return vocab


# class TextDataset(Dataset):
#     def __init__(self, df, vocab):
#         self.vocab = vocab
#         self.data = df.dropna(subset=["text"])
#         self.texts = self.data["text"].apply(tokenizer).apply(vocab)
#         self.labels = self.data["label"].astype(int)

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         return torch.tensor(self.texts.iloc[idx]), torch.tensor(self.labels.iloc[idx])

