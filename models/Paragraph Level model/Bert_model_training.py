
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
