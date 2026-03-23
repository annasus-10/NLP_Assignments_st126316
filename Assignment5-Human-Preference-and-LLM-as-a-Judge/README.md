# Assignment 5: Optimization with Human Preference & LLM-as-a-Judge

---

# Overview

This assignment focuses on two key aspects of modern Large Language Model (LLM) development:

1. **Model Alignment** – Fine-tuning a pretrained language model using human preference data to improve factuality and reduce hallucinations.
2. **Evaluation** – Implementing an **LLM-as-a-Judge** pipeline to compare the performance of a base model and an aligned model.

The training process uses **Direct Preference Optimization (DPO)**, which allows a model to learn from preferred and rejected responses without training a separate reward model.

The evaluation process uses **AlpacaEval prompts** and a strong external LLM acting as an automatic judge.

---

# Repository Structure

```
.
├── assignment5_dpo.ipynb
├── README.md
└── results.csv
```

`assignment5_dpo.ipynb`
Main notebook containing dataset preparation, DPO training, model upload, and evaluation.

`results.csv`
Table containing evaluation samples and judge verdicts.

---

# Task 1 — Dataset Preparation

We used the dataset:

```
jondurbin/truthy-dpo-v0.1
```

This dataset is designed to train models to prefer truthful answers.

Each sample contains:

| Field    | Description                        |
| -------- | ---------------------------------- |
| prompt   | User question                      |
| chosen   | Correct factual response           |
| rejected | Hallucinated or incorrect response |

Example structure:

```
{
 "prompt": "...",
 "chosen": "...",
 "rejected": "..."
}
```

The dataset was loaded using the HuggingFace `datasets` library.

---

# Task 2 — Training with Direct Preference Optimization

A pretrained instruction-tuned model was fine-tuned using **Direct Preference Optimization (DPO)**.

## Base Model

```
Qwen/Qwen2.5-0.5B-Instruct
```

This model was selected because:

* It is lightweight
* It supports instruction following
* It can be fine-tuned efficiently

## Training Approach

Training used:

* **DPOTrainer** from the `trl` library
* **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning
* **4-bit quantization** to reduce memory usage

This allowed training to run on limited hardware.

## Training Configuration

Important hyperparameters:

```
epochs = 3
learning_rate = 5e-7
batch_size = 1
lr_scheduler = linear
warmup_ratio = 0.1
```

LoRA configuration:

```
rank = 64
alpha = 128
dropout = 0.05
target_modules = [q_proj, k_proj, v_proj]
```

The model was trained to maximize the probability of the **preferred (chosen) answer** relative to the **rejected answer**.

---

# Task 3 — Model Upload

After training, the model was uploaded to the HuggingFace Hub.

Model link:

```
https://huggingface.co/annasus10/qwen2.5-0.5b-truthy-dpo
```

The uploaded repository includes:

* LoRA adapter weights
* tokenizer files
* configuration files

This allows the model to be reused or evaluated later.

---

# Task 4 — Evaluation with LLM-as-a-Judge

To evaluate whether DPO improved the model, we implemented an **LLM-as-a-Judge pipeline**.

The evaluation dataset used was:

```
tatsu-lab/alpaca_eval
```

Only the **helpful_base subset** was used.

---

# Evaluation Pipeline

The evaluation process consisted of the following steps:

1. Load AlpacaEval dataset
2. Filter to the **helpful_base** subset
3. Randomly sample **15 prompts**
4. Generate responses from:

   * Base Model
   * DPO Model
5. Send both responses to an external **judge LLM**
6. Record the verdict
7. Compute win rate

---

# Judge Prompt

The judge LLM was instructed using the following template:

```
You are a highly qualified and impartial judge evaluating two AI models.

User Instruction: {instruction}

Model A (Base Model): {base answer}

Model B (DPO Model): {dpo answer}

Evaluate both models. Output ONLY one of:
Model A
Model B
Tie
```

The judge compared both responses and returned the better one.

---

# Evaluation Results

Example result table:

| Sample | Instruction                           | Winner  |
| ------ | ------------------------------------- | ------- |
| 1      | Explain quantum mechanics simply      | Model B |
| 2      | Write a Python function for factorial | Model A |
| 3      | What are the causes of climate change | Tie     |
| ...    | ...                                   | ...     |

---

# Win Rate Calculation

The assignment specifies the following formula:

```
Win Rate =
(Model B Wins + 0.5 × Ties)
--------------------------------
Total Valid Evaluations
× 100
```

Where:

* **Model B** = DPO model
* **Model A** = Base model

---

# Example Result

```
Model B wins: 7
Ties: 3
Total evaluations: 15
```

Win rate:

```
Win Rate = (7 + 0.5×3) / 15 × 100
= 56.67%
```

This indicates that the DPO model performed slightly better than the base model on this evaluation subset.

---

# Discussion

The results suggest that **Direct Preference Optimization improved the model's helpfulness and factual accuracy in several cases**.

However, the improvement is not universal. Some prompts still favored the base model or resulted in ties.

Possible reasons include:

* Small evaluation sample size
* Limited training dataset
* Variability in LLM judge decisions

Nevertheless, the DPO model demonstrates promising improvements in alignment.

---

# Conclusion

In this assignment we implemented:

* Preference-based fine-tuning using **Direct Preference Optimization**
* Parameter-efficient training with **LoRA**
* A full **LLM-as-a-Judge evaluation pipeline**
* A quantitative comparison between a base model and an aligned model

The experiment shows that alignment techniques such as DPO can improve the quality and factual reliability of language model responses.

---

# References

HuggingFace Datasets
https://huggingface.co/datasets

DPOTrainer Documentation
https://huggingface.co/docs/trl/main/dpo_trainer

AlpacaEval Benchmark
https://huggingface.co/datasets/tatsu-lab/alpaca_eval
