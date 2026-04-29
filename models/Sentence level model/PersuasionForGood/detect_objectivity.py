import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load your CSV
df = pd.read_csv("/home/somrupa/persuasion/objective_14000.csv")  # change filename as needed
df = df.dropna(subset=["text"])  # Ensure no missing text
texts = df["text"].tolist()

# Load the zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels
labels = ["objective", "subjective"]

# Store results
objective_sentences = []

for text in tqdm(texts, desc="Checking Objectivity"):
    try:
        result = classifier(text, candidate_labels=labels)
        label = result['labels'][0]
        if label == "objective":
            objective_sentences.append(text)
    except Exception as e:
        print(f"Error for text: {text[:50]}... \n{e}")

# Save to CSV
output_df = pd.DataFrame(objective_sentences, columns=["text"])
output_df.to_csv("filtered_objective_sentences.csv", index=False)
print("✅ Saved filtered objective sentences to 'filtered_objective_sentences.csv'")
