#Code for In-domain and Cross-domain testing
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


