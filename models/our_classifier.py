# #!/usr/bin/env python3
# import os
# import re
# import time
# import random
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
# from datasets import Dataset
# import torch
# from transformers import (
#     RobertaTokenizer,
#     RobertaForSequenceClassification,
#     Trainer,
#     TrainingArguments,
#     set_seed
# )

# # ---------------------------
# # CONFIG
# # ---------------------------
# MODEL_NAME = "roberta-base"
# OUTPUT_ROOT = "./runs"   # root directory for all runs (models + logs)
# RESULTS_CSV = "cross_dataset_results_our_classifier.csv"

# MAX_LEN = 512
# BATCH_SIZE = 16
# NUM_EPOCHS = 3
# LEARNING_RATE = 2e-5
# SEED = 42
# DEBUG_N_SAMPLES =2
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------------------------
# # DATASET PATHS - put your actual paths here
# # Each dataset must have 'train' and 'test' CSV paths
# # CSVs must contain columns: text, reasoning (or raw_output), label
# # ---------------------------
# DATASET_PATHS = {
#     "CMV": {
#         "train": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/CMV_train_text_reasoning.csv",
#         "test" : "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/CMV_test_text_reasoning.csv"
#     },
#     # "Synthetic": {
#     #     # update these paths if you have separate train/test files at different locations
#     #     "train": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/Synthetic_train_text_reasoning.csv",
#     #     "test" : "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/Synthetic_test_text_reasoning.csv"  # <-- change if needed
#     # },
#     "Ling": {
#         # if Ling only had one file, duplicate it or provide proper test file
#         "train": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/Ling_train_text_reasoning.csv",
#         "test" : "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/ling_test_text_reasoning.csv"  # <-- change to real test file
#     },
#     # Add more datasets here if needed, following the same pattern
# }

# # ---------------------------
# # HELPERS
# # ---------------------------
# def normalize_label_col(df, label_col="label"):
#     """Normalize label column to binary 0/1. Expects persuasive/non-persuasive variants."""
#     if label_col not in df.columns:
#         raise ValueError(f"No label column found in dataframe. Columns: {df.columns.tolist()}")
#     def map_label(x):
#         if pd.isna(x):
#             return 0
#         s = str(x).strip().lower()
#         s = re.sub(r"[\s\-_]+", "", s)
#         if s in ("persuasive","persuade","persuasion","1","true","yes","p"):
#             return 1
#         if s in ("nonpersuasive","nonpersuasion","non","0","false","no","np"):
#             return 0
#         if "persu" in s:
#             return 1
#         try:
#             v = int(s)
#             return 1 if v != 0 else 0
#         except:
#             return 0
#     df[label_col] = df[label_col].apply(map_label)
#     return df

# def load_and_prepare(path):
#     df = pd.read_csv(path, dtype=str).fillna("")

#     if "raw_output" in df.columns and "reasoning" not in df.columns:
#         df = df.rename(columns={"raw_output": "reasoning"})
#     if "Label" in df.columns and "label" not in df.columns:
#         df = df.rename(columns={"Label": "label"})

#     if "reasoning" not in df.columns or "label" not in df.columns:
#         raise ValueError(f"Missing columns in {path}")

#     df = normalize_label_col(df, "label")

#     # ⭐ reasoning-only
#     df["combined"] = df["reasoning"].astype(str)

#     # 🔍 debug mode: only 2 datapoints
#     if DEBUG_N_SAMPLES is not None:
#         df = df.head(DEBUG_N_SAMPLES)

#     return df[["combined", "label"]]


# def df_to_dataset(df):
#     """Convert pandas df with 'combined' and 'label' columns to HF Dataset."""
#     return Dataset.from_pandas(df[["combined", "label"]], preserve_index=False)

# # ---------------------------
# # TOKENIZER
# # ---------------------------
# tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# def tokenize_batch(batch):
#     return tokenizer(batch["combined"],
#                      truncation=True,
#                      padding="max_length",
#                      max_length=MAX_LEN)

# # ---------------------------
# # TRAIN / EVAL function
# # ---------------------------
# def train_and_eval(train_df, test_df, train_name, test_name, run_id=0):
#     """Train a model on train_df and evaluate on test_df. Returns metrics dict."""
#     # Prepare output dirs
#     run_name = f"{train_name}_to_{test_name}_run{run_id}_{int(time.time())}"
#     run_dir = os.path.join(OUTPUT_ROOT, run_name)
#     os.makedirs(run_dir, exist_ok=True)

#     # set seed for reproducibility
#     set_seed(SEED + run_id)

#     # datasets
#     train_ds = df_to_dataset(train_df)
#     test_ds = df_to_dataset(test_df)

#     # tokenize
#     train_ds = train_ds.map(tokenize_batch, batched=True)
#     test_ds  = test_ds.map(tokenize_batch, batched=True)

#     # format for pytorch
#     train_ds = train_ds.rename_column("label", "labels")
#     test_ds  = test_ds.rename_column("label", "labels")
#     train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#     test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

#     # model
#     model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
#     model.to(DEVICE)

#     # training args
#     training_args = TrainingArguments(
#         output_dir=run_dir,
#         num_train_epochs=NUM_EPOCHS,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         learning_rate=LEARNING_RATE,
#         eval_strategy="no",
#         save_strategy="no",
#         logging_dir=os.path.join(run_dir, "logs"),
#         logging_steps=50,
#         seed=SEED + run_id,
#         fp16=torch.cuda.is_available()
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         tokenizer=tokenizer,
#     )

#     # train
#     trainer.train()

#     # save model & tokenizer
#     model_save_dir = os.path.join(run_dir, "model")
#     os.makedirs(model_save_dir, exist_ok=True)
#     trainer.save_model(model_save_dir)
#     tokenizer.save_pretrained(model_save_dir)

#     # evaluate on test set
#     preds = trainer.predict(test_ds)
#     y_true = preds.label_ids
#     y_pred = np.argmax(preds.predictions, axis=1)

#     acc = accuracy_score(y_true, y_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
#     clf_report = classification_report(
#     y_true,
#     y_pred,
#     labels=[0, 1],  
#     target_names=["Non-Persuasive", "Persuasive"],
#     zero_division=0
# )

#     # Save predictions and report
#     out_pred_df = test_df.reset_index(drop=True).copy()
#     out_pred_df["pred_label"] = y_pred
#     out_pred_df["true_label"] = y_true
#     out_pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

#     with open(os.path.join(run_dir, "classification_report.txt"), "w") as f:
#         f.write(clf_report)
#         f.write("\n\n")
#         f.write(f"Accuracy: {acc:.6f}\nPrecision: {precision:.6f}\nRecall: {recall:.6f}\nF1: {f1:.6f}\n")

#     # Return metrics and path
#     metrics = {
#         "Train": train_name,
#         "Test": test_name,
#         "RunDir": run_dir,
#         "Accuracy": float(acc),
#         "Precision": float(precision),
#         "Recall": float(recall),
#         "F1": float(f1)
#     }
#     return metrics

# # ---------------------------
# # MAIN: load all datasets and run experiments
# # ---------------------------
# def main():
#     os.makedirs(OUTPUT_ROOT, exist_ok=True)
#     # Load dataframes (train + test) for each dataset
#     loaded = {}
#     for name, paths in DATASET_PATHS.items():
#         train_path = paths.get("train")
#         test_path  = paths.get("test")
#         if not (train_path and test_path):
#             raise ValueError(f"Dataset {name} must have both 'train' and 'test' paths in DATASET_PATHS.")
#         if not os.path.exists(train_path):
#             raise FileNotFoundError(f"{train_path} for dataset {name} not found. Update DATASET_PATHS.")
#         if not os.path.exists(test_path):
#             raise FileNotFoundError(f"{test_path} for dataset {name} not found. Update DATASET_PATHS.")

#         print(f"Loading {name} train from {train_path} ...")
#         train_df = load_and_prepare(train_path)
#         print(f"Loading {name} test from {test_path} ...")
#         test_df  = load_and_prepare(test_path)

#         print(f"{name}: train {len(train_df)} rows, test {len(test_df)} rows, label dist (train):\n{train_df['label'].value_counts().to_dict()}")
#         loaded[name] = {"train": train_df, "test": test_df}

#     results = []
#     run_counter = 0
#     # For each train -> each test (including same-domain)
#     for train_name, train_data in loaded.items():
#         for test_name, test_data in loaded.items():
#             run_counter += 1
#             print(f"\n=== Run {run_counter}: Train on {train_name} train ({len(train_data['train'])} rows), Test on {test_name} test ({len(test_data['test'])} rows) ===")
#             metrics = train_and_eval(train_data['train'], test_data['test'], train_name, test_name, run_id=run_counter)
#             print(f"Result: Acc={metrics['Accuracy']:.4f} F1={metrics['F1']:.4f} saved to {metrics['RunDir']}")
#             results.append(metrics)

    
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(RESULTS_CSV, index=False)
#     print(f"\nAll runs finished. Summary saved to {RESULTS_CSV}")

# if __name__ == "__main__":
#     main()

# #Code  to get the count dataset summary which include file,n_rows,persuasive,non_persuasive,unknown_labels,unique_raw_labels,reasoning_missing,reasoning_avg_len,reasoning_max_len
# # import os, re, pandas as pd, numpy as np

# # # ---------- EDIT: set your files here ----------
# # FILES = {
# #     "CMV_train": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/CMV_train_text_reasoning.csv",
# #     "CMV_test" : "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/CMV_test_text_reasoning.csv",
# #     "Ling_train": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/Ling_train_text_reasoning.csv",
# #     "Ling_test" : "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/ling_test_text_reasoning.csv",
# #     "Synthetic_train": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/Synthetic_train_text_reasoning.csv",
# #     "Synthetic_test": "/home/somrupa/persuasion/RecruitmentScam/reasoning_local_mistral_outputs/reasoning/Synthetic_test_text_reasoning.csv",
# # }
# # OUT_SUMMARY = "datasets_summary_quick.csv"
# # SAMPLE_ROWS = 3
# # # ---------- end edit ----------

# # def safe_read(path):
# #     if not os.path.exists(path):
# #         raise FileNotFoundError(path)
# #     return pd.read_csv(path, dtype=str).fillna("")

# # def normalize_label(x):
# #     s = str(x).strip().lower()
# #     s = re.sub(r"[\s\-_]+","", s)
# #     if s in ("persuasive","persuade","persuasion","persuaded","p","1","true","yes"):
# #         return 1
# #     if s in ("nonpersuasive","nonpersuasion","non","np","0","false","no"):
# #         return 0
# #     if "persu" in s:
# #         return 1
# #     try:
# #         v = int(s)
# #         return 1 if v != 0 else 0
# #     except:
# #         return None

# # rows = []
# # for name, path in FILES.items():
# #     print(f"\n=== Inspecting {name} ===")
# #     try:
# #         df = safe_read(path)
# #     except Exception as e:
# #         print("ERROR reading file:", e)
# #         rows.append({"file": path, "error": str(e)})
# #         continue

# #     # check required columns
# #     for c in ("text","reasoning","label"):
# #         if c not in df.columns:
# #             print(f"  MISSING column: {c}  — columns present: {list(df.columns)}")
# #     # normalize labels
# #     df["label_norm"] = df["label"].apply(normalize_label)
# #     count_pos = int((df["label_norm"]==1).sum())
# #     count_neg = int((df["label_norm"]==0).sum())
# #     count_unknown = int(df["label_norm"].isna().sum())
# #     unique_raw = list(pd.unique(df["label"].astype(str)))
# #     # reasoning stats
# #     if "reasoning" not in df.columns:
# #         df["reasoning"] = ""
# #     reasoning_missing = int((df["reasoning"].astype(str).str.strip()=="").sum())
# #     reasoning_len_mean = float(df["reasoning"].astype(str).apply(len).mean())
# #     reasoning_len_max  = int(df["reasoning"].astype(str).apply(len).max())

# #     print(f"  Rows: {len(df)}")
# #     print(f"  Persuasive count (normalized)= {count_pos}")
# #     print(f"  Non-Persuasive count (normalized)= {count_neg}")
# #     print(f"  Unknown/malformed labels = {count_unknown}")
# #     print(f"  Unique raw label values (sample) = {unique_raw[:10]}")
# #     print(f"  Reasoning missing rows = {reasoning_missing}")
# #     print(f"  Reasoning avg len = {reasoning_len_mean:.1f}, max len = {reasoning_len_max}")

# #     print("\n  Example persuasive rows:")
# #     sample_p = df[df["label_norm"]==1].head(SAMPLE_ROWS)
# #     if sample_p.empty:
# #         print("   - None found")
# #     else:
# #         for _, r in sample_p.iterrows():
# #             t = (r["text"][:150]+"...") if len(r["text"])>150 else r["text"]
# #             print(f"   - text: {t}")
# #             print(f"     raw label: {r['label']}, reasoning len: {len(str(r['reasoning']))}")

# #     print("\n  Example non-persuasive rows:")
# #     sample_np = df[df["label_norm"]==0].head(SAMPLE_ROWS)
# #     if sample_np.empty:
# #         print("   - None found")
# #     else:
# #         for _, r in sample_np.iterrows():
# #             t = (r["text"][:150]+"...") if len(r["text"])>150 else r["text"]
# #             print(f"   - text: {t}")
# #             print(f"     raw label: {r['label']}, reasoning len: {len(str(r['reasoning']))}")

# #     rows.append({
# #         "file": path,
# #         "n_rows": len(df),
# #         "persuasive": count_pos,
# #         "non_persuasive": count_neg,
# #         "unknown_labels": count_unknown,
# #         "unique_raw_labels": ";".join(map(str, unique_raw))[:1000],
# #         "reasoning_missing": reasoning_missing,
# #         "reasoning_avg_len": reasoning_len_mean,
# #         "reasoning_max_len": reasoning_len_max
# #     })

# # pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)
# # print(f"\nSummary saved to: {OUT_SUMMARY}")

#Code to get the onfusion matrix 
# import pandas as pd
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt

# # path to prediction file
# pred_path = "//home/somrupa/persuasion/RecruitmentScam/runs/CMV_to_Synthetic_run2_1765531851/predictions.csv"

# df = pd.read_csv(pred_path)

# y_true = df["true_label"]
# y_pred = df["pred_label"]

# cm = confusion_matrix(y_true, y_pred)

# print(cm)

# # Plot
# plt.figure(figsize=(5,4))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=["Non-Persuasive", "Persuasive"],
#     yticklabels=["Non-Persuasive", "Persuasive"]
# )
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix: CMV → Ling")
# plt.tight_layout()
# plt.show()

