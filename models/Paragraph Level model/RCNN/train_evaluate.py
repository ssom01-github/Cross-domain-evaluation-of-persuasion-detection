#code for Modification of RCNN model using paragraph level data
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report
# import pandas as pd
# from rcnn_model import RCNN
# from utils import TextDataset, build_vocab

# # ===============================
# # 🔹 Paths to Nizerian dataset
# # ===============================
# NIZERIAN_TRAIN = "/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Nizerian_5_split/train.csv"
# NIZERIAN_VAL   = "/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Nizerian_5_split/val.csv"
# NIZERIAN_TEST  = "/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Nizerian_5_split/test.csv"

# # 🔹 Paths to phishing dataset
# PHISHING_TRAIN = "/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Nazario5_split/train.csv"
# PHISHING_VAL   = "/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Nazario5_split/val.csv"
# PHISHING_TEST  = "/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Nazario5_split/test.csv"


# # Load Data

# train_Nizerian = pd.read_csv(NIZERIAN_TRAIN)
# val_Nizerian   = pd.read_csv(NIZERIAN_VAL)
# test_Nizerian  = pd.read_csv(NIZERIAN_TEST)

# train_phishing = pd.read_csv(PHISHING_TRAIN)
# val_phishing   = pd.read_csv(PHISHING_VAL)
# test_phishing  = pd.read_csv(PHISHING_TEST)


# #  Build vocab (from Nizerian train set for consistency)

# vocab = build_vocab(train_Nizerian)
# PAD_IDX = vocab["<pad>"]


# # Device setup

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters

# BATCH_SIZE = 64
# EPOCHS = 5
# EMBED_DIM = 100
# HIDDEN_DIM = 128
# OUTPUT_DIM = 2


# #  Collate Function

# def collate_batch(batch):
#     texts, labels = zip(*batch)
#     padded = nn.utils.rnn.pad_sequence(texts, padding_value=PAD_IDX, batch_first=True)
#     return padded.to(device), torch.tensor(labels).to(device)


# #  DataLoaders
# # Nizerian loaders
# train_loader_r = DataLoader(TextDataset(train_Nizerian, vocab), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
# val_loader_r   = DataLoader(TextDataset(val_Nizerian, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
# test_loader_r  = DataLoader(TextDataset(test_Nizerian, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# # Phishing loaders
# train_loader_p = DataLoader(TextDataset(train_phishing, vocab), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
# val_loader_p   = DataLoader(TextDataset(val_phishing, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
# test_loader_p  = DataLoader(TextDataset(test_phishing, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# #  Training / Evaluation Functions

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


# #  Nizerian → Nizerian + Phishing

# # print("\n🔹 Training on Nizerian (train set)")
# # model = RCNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX).to(device)
# # optimizer = optim.Adam(model.parameters())
# # criterion = nn.CrossEntropyLoss()

# # for epoch in range(EPOCHS):
# #     loss = train(model, train_loader_r)
# #     print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

# # print("\n Validation on Nizerian")
# # print(evaluate(model, val_loader_r))

# # print("\n Test on Nizerian")
# # print(evaluate(model, test_loader_r))

# # print("\n Cross-domain: Test on Phishing Emails")
# # print(evaluate(model, test_loader_p))


# #  Phishing → Phishing + Nizerian

# print("\n🔹 Training on Phishing Emails(train set)")
# model = RCNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, PAD_IDX).to(device)
# optimizer = optim.Adam(model.parameters())

# for epoch in range(EPOCHS):
#     loss = train(model, train_loader_p)
#     print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

# print("\n Validation on Phishing Emails")
# print(evaluate(model, val_loader_p))

# print("\n Test on Phishing Emails")
# print(evaluate(model, test_loader_p))

# print("\n Cross-domain: Test on Nizerian")
# print(evaluate(model, test_loader_r))


#Code for Part A: K-Fold CV within each dataset (fold-wise results + averages).
# Part B: Cross-dataset evaluation (train on one, test on all others).

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import re
import os

from rcnn_model import RCNN
from utils import TextDataset, build_vocab

# ---------------------------
# Config
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
EMBED_DIM = 100
HIDDEN_DIM = 128
OUTPUT_DIM = 2
K_FOLDS = 5

def preprocess(df):
    # Rename columns
    if "Text" in df.columns:
        df = df.rename(columns={"Text": "text"})
    if "Predicted_Label" in df.columns:
        df = df.rename(columns={"Predicted_Label": "label"})

    df["text"] = df["text"].astype(str)

    # Convert string labels to numeric
    def label_to_int(x):
        if isinstance(x, str):
            x = x.strip().lower()
            if x in ["persuasive", "persuasive(1)"]:
                return 1
            elif x in ["nonpersuasive", "non-persuasive", "non persuasive", "non-persuasive(0)"]:
                return 0
            else:
                return None  # skip unknown labels
        elif isinstance(x, (int, float)):
            return int(x)
        else:
            return None

    df["label"] = df["label"].apply(label_to_int)

    # Drop rows with unmapped labels
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["text", "label"]]





# ---------------------------
# Collate Function
# ---------------------------
def collate_batch(batch):
    texts, labels = zip(*batch)
    PAD_IDX = vocab["<pad>"]
    padded = nn.utils.rnn.pad_sequence(texts, padding_value=PAD_IDX, batch_first=True)
    return padded.to(device), torch.tensor(labels).to(device)

# ---------------------------
# Training / Evaluation
# ---------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    report = classification_report(all_labels, all_preds, digits=4, output_dict=True)
    return report

# ---------------------------
# Part A: K-Fold CV
# ---------------------------
# def run_kfold_cv(df, dataset_name):
#     X = df["text"].astype(str).values
#     y = df["label"].astype(int).values
#     skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

#     accs, f1s = [], []

#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#         print(f"\n===== {dataset_name}: Fold {fold+1}/{K_FOLDS} =====")

#         train_df = df.iloc[train_idx].reset_index(drop=True)
#         val_df   = df.iloc[val_idx].reset_index(drop=True)

#         global vocab
#         vocab = build_vocab(train_df)

#         train_loader = DataLoader(TextDataset(train_df, vocab), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
#         val_loader   = DataLoader(TextDataset(val_df, vocab), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

#         model = RCNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, vocab["<pad>"]).to(device)
#         optimizer = optim.Adam(model.parameters())
#         criterion = nn.CrossEntropyLoss()

#         for epoch in range(EPOCHS):
#             loss = train(model, train_loader, optimizer, criterion)
#             print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

#         report = evaluate(model, val_loader)
#         acc = report["accuracy"]
#         f1 = report["1"]["f1-score"]

#         print(f"Fold {fold+1}: Accuracy: {acc:.4f}, F1: {f1:.4f}")
#         accs.append(acc)
#         f1s.append(f1)

#         #  Save fold-wise results
#         fold_result = pd.DataFrame({
#             "Dataset": [dataset_name],
#             "Fold": [fold+1],
#             "Accuracy": [acc],
#             "F1": [f1]
#         })
#         fold_result.to_csv("results_rcnn_folds.csv", mode="a", index=False,
#                            header=not os.path.exists("results_rcnn_folds.csv"))

#     #  Save dataset-wise average + std
#     avg_summary = pd.DataFrame({
#         "Dataset": [dataset_name],
#         "Accuracy_Mean": [np.mean(accs)],
#         "Accuracy_Std": [np.std(accs)],
#         "F1_Mean": [np.mean(f1s)],
#         "F1_Std": [np.std(f1s)]
#     })
#     avg_summary.to_csv("results_rcnn_summary.csv", mode="a", index=False,
#                        header=not os.path.exists("results_rcnn_summary.csv"))

# ---------------------------
# Part B: Cross-Dataset Evaluation
# ---------------------------
def run_cross_dataset(datasets):
    results = []

    # for train_name, train_df in datasets.items():
    #     print(f"\n\n===== Training on {train_name} =====")
    #     global vocab
    #     vocab = build_vocab(train_df)

    #     train_loader = DataLoader(TextDataset(train_df, vocab),
    #                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    #     model = RCNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, vocab["<pad>"]).to(device)
    #     optimizer = optim.Adam(model.parameters())
    #     criterion = nn.CrossEntropyLoss()

    #     # Train model
    #     for epoch in range(EPOCHS):
    #         loss = train(model, train_loader, optimizer, criterion)
    #         print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

    #     # Evaluate on all other datasets
    #     for test_name, test_df in datasets.items():
    #         if test_name == train_name:
    #             continue
    #         print(f"\n📊 Cross-domain: Test on {test_name}")
    #         test_loader = DataLoader(TextDataset(test_df, vocab),
    #                                  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
    #         report = evaluate(model, test_loader)
    #         print("Type of report:", type(report))
    #         print("Report content:", report)
    #         accuracy = report['accuracy']
    #         precision = report['weighted avg']['precision']
    #         recall = report['weighted avg']['recall']
    #         f1 = report['weighted avg']['f1-score']

    #         # Append only Accuracy and F1
    #         results.append({
    #             "Train": train_name,
    #             "Test": test_name,
    #             "Accuracy": accuracy,
    #             "Precision": precision,
    #             "Recall": recall,
    #             "F1": f1
    #         })

    # # Save all results to CSV
    # df_results = pd.DataFrame(results)
    # df_results.to_csv("cross dtaset result(RCNN).csv", index=False)
    # print(" Saved cross-dataset results to cross dtaset result(RCNN).csv")

    cross_pairs = [
        ("Ling", ["CMV", "Synthetic"]),
        ("CMV", ["Ling", "Synthetic"]),
        ("Synthetic", ["Ling", "CMV"])
    ]

    for train_name, test_list in cross_pairs:
        print(f"\n\n===== Training on {train_name} =====")
        global vocab
        train_df = datasets[train_name]
        vocab = build_vocab(train_df)

        train_loader = DataLoader(TextDataset(train_df, vocab),
                                  batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

        model = RCNN(len(vocab), EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM, vocab["<pad>"]).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(EPOCHS):
            loss = train(model, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

        # Test on specified datasets only
        for test_name in test_list:
            print(f"\n📊 Cross-domain: Test on {test_name}")
            test_df = datasets[test_name]
            test_loader = DataLoader(TextDataset(test_df, vocab),
                                     batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
            report = evaluate(model, test_loader)

            accuracy = report['accuracy']
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']

            results.append({
                "Train": train_name,
                "Test": test_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1
            })

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("cross_dataset_results_RCNN.csv", index=False)
    print(" Saved cross-dataset results to cross_dataset_results_RCNN.csv")
# --------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Load datasets
    datasets = {
        "Ling": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Ling.csv")),
        "CMV": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/CMV/processed_split/test.csv")),
        "Synthetic": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/train.csv")),
    }

    #  Check label distribution BEFORE training
    for name, df in datasets.items():
        print(f"{name} label distribution:\n", df['label'].value_counts(), "\n")

    # Part B: Cross-Dataset
    run_cross_dataset(datasets)

