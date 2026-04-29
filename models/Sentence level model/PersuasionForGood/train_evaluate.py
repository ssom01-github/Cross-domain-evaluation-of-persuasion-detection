#code implementing the persuason method using sentence level data
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
# from rcnn_model import RCNN
# from utils import TextDataset, build_vocab
# import pandas as pd
# import os

# # Paths to data
# P4G_PATH = "/home/somrupa/persuasion/persuasion_dataset.csv"
# QUORA_PATH = "/home/somrupa/persuasion/data/quora.csv"
# PERSENTSE_PATH = "/home/somrupa/persuasion/data/persentSE.csv"

# # Load Data
# df_p4g = pd.read_csv(P4G_PATH)
# df_quora = pd.read_csv(QUORA_PATH)
# df_persentse = pd.read_csv(PERSENTSE_PATH)

# # Load CSVs into DataFrames
# train_df = pd.read_csv("/home/somrupa/persuasion/persuasion_dataset.csv")
# val_df = pd.read_csv("/home/somrupa/persuasion/val_unbalanced.csv")
# test_df = pd.read_csv("/home/somrupa/persuasion/test_unbalanced.csv")


# # Build vocab only from train data
# vocab = build_vocab(train_df)
# PAD_IDX = vocab["<pad>"]

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# BATCH_SIZE = 64
# EPOCHS = 5
# EMBED_DIM = 100
# HIDDEN_DIM = 128
# OUTPUT_DIM = 2

# def collate_batch(batch):
#     texts, labels = zip(*batch)
#     lengths = [len(x) for x in texts]
#     padded = nn.utils.rnn.pad_sequence(texts, padding_value=PAD_IDX, batch_first=True)
#     return padded.to(device), torch.tensor(labels).to(device)

# # Loaders
# train_loader = DataLoader(TextDataset(train_df, vocab), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
# val_loader = DataLoader(TextDataset(val_df, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
# test_loader = DataLoader(TextDataset(test_df, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# # Cross-domain loaders
# quora_loader = DataLoader(TextDataset(df_quora, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
# persentse_loader = DataLoader(TextDataset(df_persentse, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# # Model
# model = RCNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX).to(device)
# optimizer = optim.Adam(model.parameters())
# criterion = nn.CrossEntropyLoss()

# # Training Loop
# def train(model, loader):
#     model.train()
#     total_loss = 0
#     for x, y in loader:
#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(loader)

# # Evaluation
# def evaluate(model, loader):
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for x, y in loader:
#             out = model(x)
#             preds = out.argmax(dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(y.cpu().numpy())
#     return classification_report(all_labels, all_preds, digits=4)

# # === Train Only on PersuasionForGood ===
# print("\n🔹 Training on PersuasionForGood")
# for epoch in range(EPOCHS):
#     loss = train(model, train_loader)
#     print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

# # === Evaluation ===
# print("\n Validation on PersuasionForGood")
# print(evaluate(model, val_loader))

# print("\n Test on PersuasionForGood")
# print(evaluate(model, test_loader))

# print("\n Cross-Domain Eval on Quora")
# print(evaluate(model, quora_loader))

# print("\n Cross-Domain Eval on PerSentSE")
# print(evaluate(model, persentse_loader))


