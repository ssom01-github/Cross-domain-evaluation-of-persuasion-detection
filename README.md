# Cross-domain-evaluation-of-persuasion-detection

#  Persuasion Detection in Long Documents (Cross-Domain Analysis)

##  Overview

This research project focuses on **persuasion detection in long-form text (paragraph-level)** and investigates the **limitations of existing sentence-level models**, particularly in **cross-domain generalization**.

While most prior work treats persuasion detection as a **sentence-level classification task**, this study demonstrates that:

* Persuasion is inherently **contextual**
* Models trained on one domain **fail to generalize** to others

---

##  Objectives

* Evaluate existing persuasion detection models on **multiple domains**
* Analyze **sentence-level vs paragraph-level** effectiveness
* Study **cross-domain generalization failure**
* Incorporate **LLM-based reasoning signals**
* Propose and validate **domain-mixing as a solution**

---
##  Sentence Level Datasets Used
* PersuasionForGood
*  Quora-insincere-questions: Consist approximately 1.3M  training sample with Questions are marked as 0 (Sincere) and 1 (Insincere) Link: https://www.kaggle.com/datasets/arnavsharma45/quora-insincere-questions-dataset
* PersentSe: Persuasion Sentences in Spam Email (Total sentences: 1,075 , Persuasive sentences: 216 (20.1%) Link: https://zenodo.org/records/14585764)

##  Paragraph Level Datasets Used

###  Persuasion Datasets

* CMV (ChangeMyView) – long-form argumentative discussions

###  Phishing / Scam Datasets

* Zenodo Phishing Dataset
* Recruitment Scam Dataset

### Synthetic Data

* Generated to simulate persuasion patterns across domains

---

##  Models Evaluated

This work **reproduces and benchmarks** multiple existing models:

* BERT
* RoBERTa
* RCNN
* PersuGPT
* PCoT (Persuasion-Augmented Chain-of-Thought)

---

##  Experimental Setup

### 1. Sentence-Level vs Paragraph-Level

* Initial experiments used sentence-level datasets
* Observed poor performance in capturing persuasion
* Transitioned to **paragraph-level modeling**

---

### 2. Input Variants

We evaluated three input formats:

* **Text only**
* **Reasoning only (LLM-generated features)**
* **Text + Reasoning**

 **Key Finding:**

> Combining **text + reasoning** significantly improves performance

---

### 3. In-Domain Evaluation

Models were trained and tested on same domains:

* Train on CMV → Test on CMV
* Train on Synthetic → Test on Synthetic
* Train on Ling → Test on Ling

### 4. Cross-Domain Evaluation

Models were trained and tested across different domains:

* Train on CMV → Test on Ling
* Train on Synthetic → Test on CMV
* Train on CMV → Test on Synthetic

 **Observation:**

> Models fail to generalize across domains

---

##  Results Summary

* Best F1 observed when:

  * Training and testing on similar domains
* Severe performance drop in **cross-domain settings**
* Text + Reasoning outperforms standalone inputs
* Domain-specific features strongly influence performance

---

##  Problem Identified: Cross-Domain Failure

Models trained on one dataset **do not transfer well** to others due to:

* Linguistic differences
* Variation in persuasion style
* Domain-specific intent

---

##  5-Dimension Analysis Framework

To explain cross-domain failure, persuasion was analyzed across:

1. **Emotion**
2. **Factuality**
3. **Call-To-Action (CTA)**
4. **Persuasion Technique**
5. **Politeness**

###  Key Insight:

* **CMV** → reasoning-heavy, polite
* **Marketing / Ling datasets** → CTA-heavy, action-driven

  <p align="center">
  <img src="images/Cta_CDF_ling_mistral_vs_cmv_mistral.png" width="850"/>
</p>

<p align="center">
  <em>Figure: Call-to-action CDF for Ling Vs CMV data.</em>
</p>

 These differences explain poor generalization

---

##  Manual Annotation Study

* Annotated CMV dataset manually using custom persuasion definitions
* Inter-annotator agreement:

  * **~72% agreement (360/500 samples)**
  * Cohen’s Kappa ≈ 0.4

 Indicates:

* Persuasion detection is **subjective**
* Dataset inconsistency affects model performance

---

##  Proposed Solution: Domain Mixing

### Approach:

* Train on **mixed-domain data**
* Evaluate on:

  * Mixed domain
  * Individual domains

###  Result:

> Domain mixing improves generalization across datasets

---

##  Proposed Architecture

The final model setup uses:

* Input:

  ```
  [CLS] text [SEP] reasoning
  ```
* Encoder: Fine-tuned **RoBERTa**
* Classification Head:

  * Linear layer → Softmax (Persuasive / Non-Persuasive)

---

##  Key Contributions

* Demonstrated limitations of **sentence-level persuasion detection**
* Performed **cross-domain benchmarking** across multiple datasets
* Introduced **LLM-based reasoning augmentation**
* Proposed **5-dimension framework** to explain domain gaps
* Validated **domain mixing strategy** for improved generalization
* Conducted **manual annotation study** for dataset reliability

---

##  Project Structure

```
project/
│
├── Paragraph level dataset/                # Datasets (or links)
├── Sentence level dataset/  
├── models/              # Model implementations
├── experiments/         # Training + evaluation scripts
├── utils/               # Helper functions
├── results/             # Metrics, plots, outputs
├── requirements.txt
└── README.md
```

---

##  How to Run

```bash
git clone <your-repo-link>
cd project

pip install -r requirements.txt

```

---

##  Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

##  Future Work

* Improve cross-domain robustness using domain adaptation
* Better annotation strategies for persuasion
* Incorporate structured reasoning (chain-of-thought) more effectively
* Explore transformer-based architectures for long documents

---

##  Conclusion

This project highlights a critical issue in persuasion detection:

> **Models do not generalize across domains due to differences in persuasion style and linguistic patterns.**

By incorporating reasoning signals and domain mixing, we move closer to building **robust, domain-independent persuasion detection systems**.

---
