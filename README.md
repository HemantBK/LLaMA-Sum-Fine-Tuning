# 📚 LLaMA 3.2 1B Summarization Project

## ✨ Overview

This project fine-tunes Meta's **LLaMA 3.2 1B** model using the **CNN/DailyMail**-style summarization dataset. The objective is to create a compact and accurate summarizer capable of generating high-quality summaries from lengthy articles. All training, evaluation, and comparison have been conducted locally using **Ollama** for LLM serving and **Hugging Face TRL** for training.

---

## 📊 Dataset & Preprocessing

* Dataset: CNN/DailyMail
* Format: Each record has two fields: `article` and `highlights`
* Used 1000 examples for training and 100 for testing.
* We processed the data into a list of dictionaries:

```json
[
  {"article": ..., "summary": ...},
  ...
]
```

* Saved as `cnn_articles_summaries_truncated.json`.

---

## 🧠 Objective

To fine-tune the **Meta LLaMA 3.2 1B** model on summarization using **Supervised Fine-Tuning (SFT)** where the model learns from `(input → expected summary)` pairs.

---

## 🛠️ Model & Training Configuration

* **Base Model**: `meta-llama/Llama-3.2-1B`
* **Model Type**: Causal Language Model (CAUSAL\_LM)
* **Trainer**: `trl.SFTTrainer`
* **Quantization**: 4-bit (BitsAndBytes)
* **LoRA (Low-Rank Adaptation)**: Used to enable Parameter-Efficient Fine-Tuning (PEFT)

### 📊 Full Pipeline

#### 1. Dataset Loading & Formatting

* Loaded from local JSON
* Transformed using **LLaMA-style chat templates** with roles:

  * `system`, `user`, `assistant`

#### 2. Tokenization

* Used `AutoTokenizer`
* Enabled chat-template formatting via `apply_chat_template()`

#### 3. Model Loading (Quantized)

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

#### 4. Model Preparation

* Enabled `gradient_checkpointing`
* Used `prepare_model_for_kbit_training()`

#### 5. LoRA Configuration

```python
LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

#### 6. Training

* 5 Epochs
* Batch Size: 4
* Checkpoint saved every 500 steps

#### 7. Saving Results

```python
trainer.save_model("complete_checkpoint_summarization")
trainer.model.save_pretrained("final_model_summarization")
```

---

## 📈 Models Compared

* `llama3.2:1b` (Original)
* `newmod` (Fine-tuned version 1)
* `newone` (Fine-tuned version 2)
* `newtwo` (Fine-tuned version 3)

---

## 🔬 Evaluation Metrics

* Evaluated with:

  * **ROUGE-1**, **ROUGE-2**, **ROUGE-L** (Recall-based)
  * **BLEU** (Precision-based)
  * **METEOR** (Semantic and synonym-aware)

### 🔍 Final Scores

#### ⚡ ROUGE

```
ROUGE1:
  Original  : 0.2540
  Newmod    : 0.2945
  Newone    : 0.3067
  Newtwo    : 0.3093

ROUGE2:
  Original  : 0.1020
  Newmod    : 0.1174
  Newone    : 0.1175
  Newtwo    : 0.1436

ROUGEL:
  Original  : 0.1792
  Newmod    : 0.2079
  Newone    : 0.2023
  Newtwo    : 0.2219
```

#### ⚡ BLEU

```
BLEU:
  Original  : 0.0485
  Newmod    : 0.0553
  Newone    : 0.0551
  Newtwo    : 0.0649
```

#### ⚡ METEOR

```
METEOR:
  Original  : 0.3398
  Newmod    : 0.3907
  Newone    : 0.3933
  Newtwo    : 0.4075
```

---

## 🔍 Insights

* **Higher scores** in all metrics = better summarization quality.
* `newtwo` performed **best overall**:

  * Highest ROUGE-2 and ROUGE-L: more accurate and fluent.
  * Highest BLEU: more word match precision.
  * Highest METEOR: better semantic alignment.

---


## 🚀 Summary

This project demonstrates:

* Efficient fine-tuning with **LoRA** + **Quantization**
* Use of **chat-style inputs** for LLaMA training
* Full-stack pipeline from preprocessing → training → evaluation
* Comparison using multiple summarization metrics

Perfect for demonstrating summarization capability, fine-tuning expertise, and model evaluation in interviews or portfolio projects.

---

## 🔗 File Structure

```
.
├── train.py
├── eval.py
├── new_eval.py
├── cnn_articles_summaries_truncated.json
├── test-10.json
├── comparison_output.json
├── README.md
```




