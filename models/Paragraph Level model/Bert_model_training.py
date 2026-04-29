#code for processing CMV pair_task data 
# import pandas as pd
# import re
# from bs4 import BeautifulSoup
# from sklearn.model_selection import train_test_split
# import os

# # ---------------------------
# # Paths
# # ---------------------------
# TRAIN_SRC = "/home/somrupa/persuasion/RecruitmentScam/New_Dataset/ChangeMyView/cmv/pair_task/train_pair_data.jsonlist/train_pair_data.csv"
# HELDOUT_SRC = "/home/somrupa/persuasion/RecruitmentScam/New_Dataset/ChangeMyView/cmv/pair_task/heldout_pair_data.jsonlist/heldout_pair_data.csv"
# OUT_DIR = "/home/somrupa/persuasion/RecruitmentScam/CMV/processed_split"
# os.makedirs(OUT_DIR, exist_ok=True)

# # ---------------------------
# # Helpers
# # ---------------------------
# def clean_text(x):
#     if pd.isna(x):
#         return ""
#     text = BeautifulSoup(str(x), "html.parser").get_text(separator=" ")
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def extract_replies(df):
#     rows = []
#     for col in df.columns:
#         if col.startswith("positive/comments") and col.endswith("/body"):
#             for txt in df[col].dropna():
#                 rows.append((clean_text(txt), 1))
#         elif col.startswith("negative/comments") and col.endswith("/body"):
#             for txt in df[col].dropna():
#                 rows.append((clean_text(txt), 0))
#     return pd.DataFrame(rows, columns=["text", "label"])

# # ---------------------------
# # Load both CSVs
# # ---------------------------
# df_train = pd.read_csv(TRAIN_SRC, low_memory=False)
# df_heldout = pd.read_csv(HELDOUT_SRC, low_memory=False)

# print("Train_pair shape:", df_train.shape)
# print("Heldout_pair shape:", df_heldout.shape)

# # ---------------------------
# # Extract persuasive vs non-persuasive replies
# # ---------------------------
# flat_train = extract_replies(df_train)
# flat_heldout = extract_replies(df_heldout)

# combined = pd.concat([flat_train, flat_heldout], ignore_index=True)
# combined = combined[combined["text"].str.strip() != ""]  # drop empties

# print("Combined dataset size:", len(combined))
# print(combined["label"].value_counts())

# # ---------------------------
# # Stratified 70/10/20 split
# # ---------------------------
# train, temp = train_test_split(combined, test_size=0.3, stratify=combined["label"], random_state=42)
# val, test   = train_test_split(temp, test_size=2/3, stratify=temp["label"], random_state=42)

# # ---------------------------
# # Save splits
# # ---------------------------
# for name, split in [("train", train), ("val", val), ("test", test)]:
#     split.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
#     print(f"\n{name.upper()} set: {len(split)} samples")
#     print(split['label'].value_counts())

# print(f"\n✅ Done! Saved processed splits in {OUT_DIR}")

#Code for merging train/test / val set into single dataset 
# Load your CMV splits
# import pandas as pd
# import re
# cmv_train = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/New_Dataset/CMV_split/train.csv")
# cmv_test = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/New_Dataset/CMV_split/test.csv")
# cmv_val = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/New_Dataset/CMV_split/val.csv")
# def preprocess(df):
#     df = df.rename(columns={"Text": "text", "Predicted_Label": "label"})
#     df["text"] = df["text"].astype(str)

#     def normalize_label(x):
#         x = str(x).strip().lower()
#         x = re.sub(r"[\s\-_]+", "", x)
#         return 1 if x == "persuasive" else 0

#     df["label"] = df["label"].apply(normalize_label)
#     return df[["text", "label"]]

# # Concatenate into one dataset
# cmv_full = pd.concat([cmv_train, cmv_test, cmv_val], ignore_index=True)

# # Preprocess into Persuasive / Non-Persuasive format
# cmv_full = preprocess(cmv_full)
# cmv_full.to_csv("/home/somrupa/persuasion/RecruitmentScam/New_Dataset/CMV_split/CMV_full.csv", index=False)

# print("CMV dataset size:", len(cmv_full))
# print("CMV full dataset saved at: /home/somrupa/persuasion/RecruitmentScam/New_Dataset/CMV_split/CMV_full.csv")

# # code for spitting the data befor training the bert model and dropped 6 rows due to unmapped labels,
# # total we have Dropped 6 rows due to unmapped labels
# #  Saved: train=2447, val=306, test=306
import pandas as pd
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import re
import os


SRC = "/home/somrupa/persuasion/Persuasion_SOM/Paragraph_DATASET/CMV/CMV2.csv"  #data from phising data "ling" with persuasion label
OUT_DIR = "/home/somrupa/persuasion/Persuasion_SOM/Paragraph_DATASET"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SRC)   # columns: Text, Predicted_Label
print("Unique raw labels:", df["label"].unique())

# 1) Clean HTML only
def clean_html(x):
    if pd.isna(x): 
        return ""
    return BeautifulSoup(str(x), "html.parser").get_text(separator=" ").strip()

df["text"] = df["text"].apply(clean_html)

# 2) Normalize labels
def normalize_label(x):
    if pd.isna(x):
        return None
    x = str(x).strip().lower()
    x = re.sub(r"[\s\-_]+", "", x)  # remove spaces, dashes, underscores
    return x

df["norm_label"] = df["label"].apply(normalize_label)

# 3) Map normalized labels to binary
label_map = {"persuasive": 1, "nonpersuasive": 0}
df["label"] = df["norm_label"].map(label_map)

# 4) Drop NaN labels (unmapped)
before = len(df)
df = df.dropna(subset=["label"])
after = len(df)
print(f"Dropped {before - after} rows due to unmapped labels")

# 5) Train/Val/Test split
train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val, test   = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

# 6) Save
for name, split in [("train",train),("val",val),("test",test)]:
    split[["text","label"]].to_csv(f"{OUT_DIR}/{name}.csv", index=False)

print(f"✅ Saved: train={len(train)}, val={len(val)}, test={len(test)}")

#Code for  class distribution  for train as 70% , test 20%, and val 10%
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from bs4 import BeautifulSoup
# import os

# # -------------------------------
# # Paths
# # -------------------------------
# SRC = "/home/somrupa/persuasion/RecruitmentScam/SyntheticData/synthetic_standardized.csv"
# OUT_DIR = "/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/"
# os.makedirs(OUT_DIR, exist_ok=True)

# # -------------------------------
# # Load CSV
# # -------------------------------
# df = pd.read_csv(SRC)
# print("Unique raw labels:", df["Predicted_Label"].unique())

# # -------------------------------
# # 1) Clean HTML from Text column
# # -------------------------------
# def clean_html(x):
#     if pd.isna(x): 
#         return ""
#     return BeautifulSoup(str(x), "html.parser").get_text(separator=" ").strip()

# df["text"] = df["Text"].apply(clean_html)

# # -------------------------------
# # 2) Use numeric labels directly
# # -------------------------------
# df["label"] = df["Predicted_Label"]

# # Drop any rows with missing labels just in case
# before = len(df)
# df = df.dropna(subset=["label"])
# after = len(df)
# print(f"Dropped {before - after} rows due to missing labels")

# print("\nOverall dataset stats:")
# print(df["label"].value_counts().rename({0:"nonpersuasive", 1:"persuasive"}))

# # -------------------------------
# # 3) Train/Validation/Test split
# # -------------------------------
# train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
# val, test   = train_test_split(temp, test_size=2/3, stratify=temp["label"], random_state=42)

# # -------------------------------
# # 4) Save splits and print stats
# # -------------------------------
# for name, split in [("train", train), ("val", val), ("test", test)]:
#     split[["text","label"]].to_csv(os.path.join(OUT_DIR, f"{name}.csv"), index=False)
#     print(f"\n{name.upper()} set: {len(split)} samples")
#     print(split["label"].value_counts().rename({0:"nonpersuasive", 1:"persuasive"}))

# print(f"\n✅ Saved splits to {OUT_DIR}")


#CODE for BERT MODEL Training
# import pandas as pd
# import torch
# from datasets import Dataset
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# # ---------------------------
# # 1. Load train/val/test CSVs
# # ---------------------------
# train_df = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Ling_split/train.csv")
# val_df   = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Ling_split/val.csv") # 305 data points
# test_df  = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Ling_split/test.csv") # 306 data points

# # Cross-domain phishing test set
# phishing_test_df = pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Bert_Data/Ling_split/test.csv")

# print("Train columns:", train_df.columns)
# print("Val columns:", val_df.columns)
# print("Test columns:", test_df.columns)
# print("Phishing columns:", phishing_test_df.columns)

# # Ensure labels are integers
# for df in [train_df, val_df, test_df, phishing_test_df]:
#     df["label"] = df["label"].astype(int)

# # Convert to HuggingFace Datasets
# train_ds = Dataset.from_pandas(train_df, preserve_index=False)
# val_ds   = Dataset.from_pandas(val_df, preserve_index=False)
# test_ds  = Dataset.from_pandas(test_df, preserve_index=False)
# phishing_test_ds = Dataset.from_pandas(phishing_test_df, preserve_index=False)

# # ---------------------------
# # 2. Tokenizer
# # ---------------------------
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# def tokenize(batch):
#     return tokenizer(
#         list(map(str, batch["text"])),  # ensure string type
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )

# train_ds = train_ds.map(tokenize, batched=True)
# val_ds   = val_ds.map(tokenize, batched=True)
# test_ds  = test_ds.map(tokenize, batched=True)
# phishing_test_ds = phishing_test_ds.map(tokenize, batched=True)

# # Rename "label" → "labels"
# train_ds = train_ds.rename_column("label", "labels")
# val_ds   = val_ds.rename_column("label", "labels")
# test_ds  = test_ds.rename_column("label", "labels")
# phishing_test_ds = phishing_test_ds.rename_column("label", "labels")

# # Set PyTorch format
# train_ds.set_format("torch")
# val_ds.set_format("torch")
# test_ds.set_format("torch")
# phishing_test_ds.set_format("torch")

# # ---------------------------
# # 3. Model
# # ---------------------------
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# # ---------------------------
# # 4. Metrics
# # ---------------------------
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
#     acc = accuracy_score(labels, preds)
#     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# # ---------------------------
# # 5. Trainer setup
# # ---------------------------
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     load_best_model_at_end=True,
#     metric_for_best_model="f1",
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# # ---------------------------
# # 6. Training
# # ---------------------------
# trainer.train()

# # ---------------------------
# # 7. In-Domain Evaluation
# # ---------------------------
# # print("\nIn-Domain Evaluation (Nizerian_5 → Nizerian_5 test):")
# # preds_in = trainer.predict(test_ds)
# # y_true_in = preds_in.label_ids
# # y_pred_in = preds_in.predictions.argmax(-1)
# # print(classification_report(y_true_in, y_pred_in, target_names=["Non-Persuasive","Persuasive"]))

# # ---------------------------
# # # 8. Cross-Domain Evaluation
# # # ---------------------------
# print("\nCross-Domain Evaluation (Nizerian_5 → Nazario_5):")
# preds_cross = trainer.predict(phishing_test_ds)
# y_true_cross = preds_cross.label_ids
# y_pred_cross = preds_cross.predictions.argmax(-1)
# print(classification_report(y_true_cross, y_pred_cross, target_names=["Non-Persuasive","Persuasive"]))


#Extend bert code for two part
#A) k-fold cross validation for each dataset(total we have 3 dataset)

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, accuracyscore, precision_recall_fscore_support
# import re

# import torch
# from datasets import Dataset
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# # ---------------------------
# # 1. Config
# # ---------------------------
# model_name = "bert-base-uncased"
# num_labels = 2
# k_folds = 5
# batch_size = 16
# num_epochs = 3
# learning_rate = 2e-5

# # ---------------------------
# # 2. Utility functions
# # ---------------------------
# def preprocess(df):
#     """Rename and normalize labels"""
#     df = df.rename(columns={"Text": "text", "Predicted_Label": "label"})
#     df["text"] = df["text"].astype(str)

#     def normalize_label(x):
#         x = str(x).strip().lower()
#         x = re.sub(r"[\s\-_]+", "", x)  # remove spaces, dashes, underscores
#         return 1 if x == "persuasive" else 0

#     df["label"] = df["label"].apply(normalize_label)
#     return df[["text", "label"]]

# def tokenize_data(df, tokenizer):
#     dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
#     dataset = dataset.map(lambda x: tokenizer(
#         list(map(str, x["text"])),
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     ), batched=True)
#     dataset = dataset.rename_column("label", "labels")
#     dataset.set_format("torch")
#     return dataset

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
#     acc = accuracy_score(labels, preds)
#     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# def train_and_evaluate(train_df, val_df, tokenizer):
#     train_ds = tokenize_data(train_df, tokenizer)
#     val_ds   = tokenize_data(val_df, tokenizer)

#     model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

#     training_args = TrainingArguments(
#         output_dir="./results",
#         evaluation_strategy="epoch",
#         save_strategy="no",
#         learning_rate=learning_rate,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         num_train_epochs=num_epochs,
#         weight_decay=0.01,
#         logging_dir="./logs",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()
#     preds = trainer.predict(val_ds)
#     return preds

# # ---------------------------
# # 3. Load datasets
# # ---------------------------
# datasets = {
#     # "Ling": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Ling.csv")),
#     # "Nizerian_5": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Nizerian_5.csv")),
#     # "Nazario_5": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Nazario5.csv")),
#     "Trec_06": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Trec_06.csv")),
# }

# tokenizer = BertTokenizer.from_pretrained(model_name)

# # ---------------------------
# # 4. Part A: K-Fold CV within each dataset
# # ---------------------------
# for name, df in datasets.items():
#     print(f"\n\n===== {k_folds}-Fold CV on {name} Dataset =====")
#     X = df["text"].values
#     y = df["label"].values

#     skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
#     accs, f1s = [], []

#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#         print(f"\n--- Fold {fold+1} ---")
#         train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

#         preds = train_and_evaluate(train_df, val_df, tokenizer)
#         y_true, y_pred = preds.label_ids, preds.predictions.argmax(-1)

#         print(classification_report(y_true, y_pred, target_names=["Non-Persuasive", "Persuasive"]))
#         accs.append(accuracy_score(y_true, y_pred))
#         f1s.append(precision_recall_fscore_support(y_true, y_pred, average="binary")[2])

#     print(f"\nAverage Accuracy: {np.mean(accs):.4f}, Average F1: {np.mean(f1s):.4f}")

# # ---------------------------
# # 5. Part B: Cross-Dataset Evaluation(saving into csv for each of the average metrics)
# # ---------------------------
# # from sklearn.metrics import precision_recall_fscore_support

# # results = []

# # for train_name, train_df in datasets.items():
# #     for test_name, test_df in datasets.items():
# #         if train_name == test_name:
# #             continue  # skip same-domain

# #         print(f"\n\n===== Train on {train_name}, Test on {test_name} =====")
# #         preds = train_and_evaluate(train_df, test_df, tokenizer)
# #         y_true, y_pred = preds.label_ids, preds.predictions.argmax(-1)

# #         report = classification_report(y_true, y_pred, target_names=["Non-Persuasive", "Persuasive"])
# #         print(report)

# #         precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
# #         acc = accuracy_score(y_true, y_pred)

# #         results.append({
# #             "Train": train_name,
# #             "Test": test_name,
# #             "Accuracy": acc,
# #             "Precision": precision,
# #             "Recall": recall,
# #             "F1": f1
# #         })

# # df_cross = pd.DataFrame(results)
# # df_cross.to_csv("cross_dataset_results.csv", index=False)
# # print("\nCross-dataset results saved to cross_dataset_results.csv")  here is hencode for bert model shall i use this with the persigpt model?
#code for plotting the bar graph for cross dataset evaluation
# import matplotlib.pyplot as plt

# # Data
# datasets = ['Ling', 'Nizerian_5', 'Nazario_5']
# f1_scores = [0.73, 0.75, 0.75]

# # Plot
# plt.figure(figsize=(6,4))
# bars = plt.bar(datasets, f1_scores, color=['skyblue', 'lightgreen', 'salmon'])

# # Add labels above bars
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", 
#              ha='center', va='bottom', fontsize=10, fontweight='bold')

# # Titles and labels
# plt.title("Cross-Dataset Evaluation (Train on Trec_06)", fontsize=12, fontweight='bold')
# plt.ylabel("F1-Score", fontsize=11)
# plt.ylim(0, 1.0)  # keep scale between 0–1 for clarity

# plt.tight_layout()
# plt.savefig("Dataset trained on Ling", dpi=300)
# plt.show()

#code for persugpt from hgging face similar to bert with some more modification(persugpt + Lora +small batch)
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
# import re

# import torch
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# from peft import LoraConfig, get_peft_model
# import os

# # --------------------------- Config ---------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)
# if device == "cuda":
#     print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
#     print("GPU ID:", torch.cuda.current_device())

# model_name = "Chuhaojin/PersuGPT"
# num_labels = 2
# k_folds = 5
# batch_size = 2
# num_epochs = 2
# learning_rate = 2e-5

# # --------------------------- Utility Functions ---------------------------
# # def preprocess(df):
# #     df = df.rename(columns={"Text": "text", "Predicted_Label": "label"})
# #     df["text"] = df["text"].astype(str)

# #     def normalize_label(x):
# #         x = str(x).strip().lower()
# #         x = re.sub(r"[\s\-_]+", "", x)
# #         return 1 if x == "persuasive" else 0

# #     df["label"] = df["label"].apply(normalize_label)
# #     return df[["text", "label"]]

# def preprocess(df):
#     # Unify column names
#     if "Text" in df.columns:
#         df = df.rename(columns={"Text": "text"})
#     if "Predicted_Label" in df.columns:
#         df = df.rename(columns={"Predicted_Label": "label"})
#     if "label" not in df.columns:
#         raise ValueError(f"Expected a 'label' column after renaming, but got: {df.columns.tolist()}")

#     df["text"] = df["text"].astype(str)

#     def normalize_label(x):
#         x = str(x).strip().lower()
#         x = re.sub(r"[\s\-_]+", "", x)
#         return 1 if x == "persuasive" else 0

#     # Normalize only if labels are not already 0/1
#     if df["label"].dtype == object or df["label"].dtype == str:
#         df["label"] = df["label"].apply(normalize_label)

#     return df[["text", "label"]]



# def tokenize_data(df, tokenizer):
#     dataset = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
#     dataset = dataset.map(lambda x: tokenizer(
#         list(map(str, x["text"])),
#         truncation=True,
#         padding="max_length",
#         max_length=64
#     ), batched=True)
#     dataset = dataset.rename_column("label", "labels")
#     dataset.set_format("torch")
#     return dataset

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
#     acc = accuracy_score(labels, preds)
#     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# def train_and_evaluate(train_df, val_df, tokenizer):
#     train_ds = tokenize_data(train_df, tokenizer)
#     val_ds = tokenize_data(val_df, tokenizer)

#     # Load PersuGPT in 8-bit
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         num_labels=num_labels,
#         device_map="auto",
#         load_in_8bit=True,
#         torch_dtype=torch.float16
#     )

#     # Add LoRA adapters
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="SEQ_CLS"
#     )
#     model = get_peft_model(model, lora_config)

#     # Ensure padding token
#     model.config.pad_token_id = tokenizer.pad_token_id

#     training_args = TrainingArguments(
#         output_dir="./results",
#         eval_strategy="epoch",
#         save_strategy="no",
#         learning_rate=learning_rate,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         num_train_epochs=num_epochs,
#         weight_decay=0.01,
#         logging_dir="./logs",
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()
#     preds = trainer.predict(val_ds)
#     return preds

# # --------------------------- Load Dataset ---------------------------
# datasets = {
    
#     "CMV": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/CMV/processed_split/test.csv")),
#     "Synthetic": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/train.csv")),
#     "Ling": preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Ling.csv")),
# } 
# ling_raw = preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/result_of_phising_data_Ling.csv"))
# print("Ling class distribution:")
# print(ling_raw["label"].value_counts())

# cmv_raw = preprocess(pd.read_csv("//home/somrupa/persuasion/RecruitmentScam/CMV/processed_split/test.csv"))
# print("CMV class distribution:")
# print(cmv_raw["label"].value_counts())

# synthetic_raw = preprocess(pd.read_csv("/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/train.csv"))
# print("Synthetic class distribution:")
# print(synthetic_raw["label"].value_counts())

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # # --------------------------- K-Fold Cross Validation ---------------------------
# # # for name, df in datasets.items():
# # #     print(f"\n\n===== {k_folds}-Fold CV on {name} Dataset =====")
# # #     X = df["text"].values
# # #     y = df["label"].values

# # #     skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
# # #     accs, f1s = [], []

# # #     # for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
# # #     #     print(f"\n--- Fold {fold+1} ---")
# # #     #     train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

# # #     #     # Small subset for dry-run
# # #     #     train_small = train_df.sample(min(50, len(train_df)), random_state=42)
# # #     #     val_small = val_df.sample(min(50, len(val_df)), random_state=42)

# # #     #     preds = train_and_evaluate(train_small, val_small, tokenizer)
# # #     #     y_true, y_pred = preds.label_ids, preds.predictions.argmax(-1)

# # #     #     print(classification_report(y_true, y_pred, target_names=["Non-Persuasive", "Persuasive"]))
# # #     #     accs.append(accuracy_score(y_true, y_pred))
# # #     #     f1s.append(precision_recall_fscore_support(y_true, y_pred, average="binary")[2])

# # #     # print(f"\nAverage Accuracy: {np.mean(accs):.4f}, Average F1: {np.mean(f1s):.4f}")

# # #     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
# # #         print(f"\n--- Fold {fold+1} ---")
# # #         train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

# # #         preds = train_and_evaluate(train_df, val_df, tokenizer)
# # #         y_true, y_pred = preds.label_ids, preds.predictions.argmax(-1)

# # #         print(classification_report(y_true, y_pred, target_names=["Non-Persuasive", "Persuasive"]))
# # #         accs.append(accuracy_score(y_true, y_pred))
# # #         f1s.append(precision_recall_fscore_support(y_true, y_pred, average="binary")[2])

# # #     print(f"\nAverage Accuracy: {np.mean(accs):.4f}, Average F1: {np.mean(f1s):.4f}")


# #     # Part B: Cross-Dataset Evaluation(saving into csv for each of the average metrics)
# # # # ---------------------------
# from sklearn.metrics import precision_recall_fscore_support

# results = []

# for train_name, train_df in datasets.items():
#     for test_name, test_df in datasets.items():
#         if train_name == test_name:
#             continue  # skip same-domain

#         print(f"\n\n===== Train on {train_name}, Test on {test_name} =====")
#         preds = train_and_evaluate(train_df, test_df, tokenizer)
#         y_true, y_pred = preds.label_ids, preds.predictions.argmax(-1)

#         report = classification_report(
#          y_true, 
#          y_pred, 
#          labels=[0, 1], 
#          target_names=["Non-Persuasive", "Persuasive"], 
#          zero_division=0
# )

#         print(report)

#         # 🔹 Sanity check: predicted class distribution
#         unique, counts = np.unique(y_pred, return_counts=True)
#         print("Predicted class distribution:", dict(zip(unique, counts)))

#         precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
#         acc = accuracy_score(y_true, y_pred)

#         results.append({
#             "Train": train_name,
#             "Test": test_name,
#             "Accuracy": acc,
#             "Precision": precision,
#             "Recall": recall,
#             "F1": f1
#         })

# df_cross = pd.DataFrame(results)
# df_cross.to_csv("cross_dataset_results.csv(persugpt)", index=False)
# print("\nCross-dataset results saved to cross_dataset_results(persugpt).csv")




# import os
# import re
# import json
# import numpy as np
# import pandas as pd

# import torch
# from datasets import Dataset
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     Trainer,
#     TrainingArguments
# )

# from peft import LoraConfig, get_peft_model

# # =========================
# # CONFIG
# # =========================
# MODEL_NAME = "Chuhaojin/PersuGPT"
# NUM_LABELS = 2

# BATCH_SIZE = 2
# NUM_EPOCHS = 2
# LEARNING_RATE = 2e-5

# BASE_OUTPUT_DIR = "./runs_persugpt"
# os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)

# # =========================
# # DATASET PATHS 
# # =========================
# DATASETS = {
#     "CMV": {
#         "train": "/home/somrupa/persuasion/RecruitmentScam/CMV/processed_split/train.csv",
#         "val":   "/home/somrupa/persuasion/RecruitmentScam/CMV/processed_split/val.csv",
#         "test":  "/home/somrupa/persuasion/RecruitmentScam/CMV/processed_split/test.csv",
#     },
#     "Ling": {
#         "train": "home/somrupa/persuasion/RecruitmentScam/Phising_Data/Ling_split/train.csv",
#         "val":   "/home/somrupa/persuasion/RecruitmentScam/Phising_Data/Ling_split/val.csv",
#         "test":  "/home/somrupa/persuasion/RecruitmentScam/Phising_Data/Ling_split/test.csv",
#     },
#     "Synthetic": {
#         "train": "/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/train.csv",
#         "val":   "/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/val.csv",
#         "test":  "/home/somrupa/persuasion/RecruitmentScam/Synthetic_data_1/test.csv",
#     }
# }

# # =========================
# # PREPROCESSING
# # =========================
# def normalize_label(x):
#     if pd.isna(x):
#         return 0
#     try:
#         return 1 if float(x) == 1.0 else 0
#     except:
#         pass
#     s = str(x).strip().lower()
#     s = re.sub(r"[\s\-_]+", "", s)
#     return 1 if "persu" in s else 0


# def load_csv(path):
#     df = pd.read_csv(path).fillna("")
#     if "Text" in df.columns:
#         df = df.rename(columns={"Text": "text"})
#     if "Label" in df.columns:
#         df = df.rename(columns={"Label": "label"})

#     df["text"] = df["text"].astype(str)
#     df["label"] = df["label"].apply(normalize_label)

#     return df[["text", "label"]]


# # =========================
# # TOKENIZATION
# # =========================
# def tokenize_df(df, tokenizer):
#     ds = Dataset.from_pandas(df, preserve_index=False)
#     ds = ds.map(
#         lambda x: tokenizer(
#             x["text"],
#             truncation=True,
#             padding="max_length",
#             max_length=64
#         ),
#         batched=True
#     )
#     ds = ds.rename_column("label", "labels")
#     ds.set_format("torch")
#     return ds


# # =========================
# # METRICS
# # =========================
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         labels, preds, average="binary", zero_division=0
#     )
#     acc = accuracy_score(labels, preds)
#     return {
#         "accuracy": acc,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }


# # =========================
# # TRAIN MODEL (ONCE)
# # =========================
# def train_model(train_df, val_df, tokenizer, run_dir):
#     train_ds = tokenize_df(train_df, tokenizer)
#     val_ds   = tokenize_df(val_df, tokenizer)

#     model = AutoModelForSequenceClassification.from_pretrained(
#         MODEL_NAME,
#         num_labels=NUM_LABELS,
#         load_in_8bit=True,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )

#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "v_proj"],
#         lora_dropout=0.1,
#         bias="none",
#         task_type="SEQ_CLS"
#     )
#     model = get_peft_model(model, lora_config)

#     training_args = TrainingArguments(
#         output_dir=run_dir,
#         eval_strategy="epoch",
#         save_strategy="no",
#         learning_rate=LEARNING_RATE,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=NUM_EPOCHS,
#         logging_dir=os.path.join(run_dir, "logs"),
#         report_to="none"
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=val_ds,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     trainer.train()
#     return trainer


# # =========================
# # EVALUATE ON TEST SET
# # =========================
# def evaluate_model(trainer, test_df, tokenizer, run_dir, test_name):
#     test_ds = tokenize_df(test_df, tokenizer)
#     preds = trainer.predict(test_ds)

#     y_true = preds.label_ids
#     y_pred = preds.predictions.argmax(-1)

#     report = classification_report(
#         y_true,
#         y_pred,
#         labels=[0, 1], 
#         target_names=["Non-Persuasive", "Persuasive"],
#         zero_division=0
#     )

#     with open(os.path.join(run_dir, f"test_{test_name}_report.txt"), "w") as f:
#         f.write(report)

#     precision, recall, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, average="binary", zero_division=0
#     )
#     acc = accuracy_score(y_true, y_pred)

#     return {
#         "Test": test_name,
#         "Accuracy": acc,
#         "Precision": precision,
#         "Recall": recall,
#         "F1": f1
#     }


# # =========================
# # MAIN EXPERIMENT
# # =========================
# def main():
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     results = []

#     for train_name, paths in DATASETS.items():
#         print(f"\n=== Training on {train_name} ===")

#         run_dir = os.path.join(BASE_OUTPUT_DIR, train_name)
#         os.makedirs(run_dir, exist_ok=True)

#         train_df = load_csv(paths["train"])
#         val_df   = load_csv(paths["val"])

#         trainer = train_model(train_df, val_df, tokenizer, run_dir)

#         for test_name, test_paths in DATASETS.items():
#             test_df = load_csv(test_paths["test"])
#             metrics = evaluate_model(
#                 trainer, test_df, tokenizer, run_dir, test_name
#             )
#             metrics["Train"] = train_name
#             results.append(metrics)

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(
#         os.path.join(BASE_OUTPUT_DIR, "cross_dataset_results.csv"),
#         index=False
#     )
#     print("\n✅ All results saved.")


# if __name__ == "__main__":
#     main()
