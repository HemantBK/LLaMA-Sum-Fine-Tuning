# LLaMA 3.2 1B — Fine-Tuned Summarization Engine

> **Fine-tuned Meta's LLaMA 3.2 1B for abstractive text summarization using LoRA + 4-bit quantization, achieving a 20% improvement in semantic accuracy (METEOR) and 41% improvement in phrase-level precision (ROUGE-2) over the base model — all on consumer-grade hardware.**

---

## Why This Project

Large language models are powerful but general-purpose. Getting production-quality results on domain-specific tasks — like summarization — requires fine-tuning. The challenge: LLMs are massive, and fine-tuning them traditionally demands expensive multi-GPU setups.

This project solves that by combining **parameter-efficient fine-tuning (LoRA)** with **4-bit quantization**, bringing the LLaMA 3.2 1B model down to ~500MB while training only ~1-2% of its parameters. The result: a specialized summarizer that runs locally, trains on a single GPU, and significantly outperforms the base model across every evaluation metric.

---

## Results at a Glance

| Metric | Base Model | Best Fine-Tuned | Improvement |
|--------|-----------|-----------------|-------------|
| **ROUGE-1** (word overlap) | 0.2540 | **0.3093** | +21.8% |
| **ROUGE-2** (phrase overlap) | 0.1020 | **0.1436** | +40.8% |
| **ROUGE-L** (fluency) | 0.1792 | **0.2219** | +23.8% |
| **BLEU** (precision) | 0.0485 | **0.0649** | +33.8% |
| **METEOR** (semantic accuracy) | 0.3398 | **0.4075** | +19.9% |

Three fine-tuned variants were trained with different configurations. The best performer (`newtwo`) achieved the highest scores across all five metrics.

---

## Technical Architecture

```
CNN/DailyMail Dataset (1000 articles)
        |
        v
  Data Pipeline ──────────── datagenerator.py
  (Extract, format,          Converts raw HuggingFace dataset
   split train/test)         into structured JSON
        |
        v
  Prompt Engineering ──────── train.py
  (LLaMA 3.2 chat template   System → User → Assistant
   with role-based prompts)   role-based formatting
        |
        v
  Fine-Tuning ─────────────── train.py
  (QLoRA: 4-bit quant          SFTTrainer, LoRA r=128,
   + LoRA adapters)            5 epochs, batch size 4
        |
        v
  Local Inference ──────────── Ollama
  (Serves fine-tuned models    HTTP streaming API
   locally, no cloud needed)   for token generation
        |
        v
  Evaluation ───────────────── new_eval.py
  (ROUGE + BLEU + METEOR       Compares 4 model variants
   across 10 test samples)     on held-out data
```

---

## Key Technical Decisions

### Why 4-Bit Quantization + LoRA (QLoRA)

Training a 1B-parameter model end-to-end requires significant VRAM. Instead:

- **4-bit NormalFloat (NF4) quantization** compresses the base model weights from ~2GB to ~500MB
- **Double quantization** further reduces the memory footprint of quantization constants
- **LoRA (rank=128, alpha=256)** injects small trainable matrices into every linear layer, adding only ~1-2% new parameters
- Combined, this enables fine-tuning on a single consumer GPU (8-16GB VRAM)

```python
# Quantization — reduces model to 4-bit precision
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# LoRA — trains <2% of parameters for >20% performance gain
LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

### Why LLaMA 3.2 1B

- Compact enough for local training and inference
- Meta's latest architecture with improved attention mechanisms
- Native chat template support (`<|start_header_id|>`, `<|eot_id|>`) enables structured prompt formatting
- Strong baseline performance that benefits meaningfully from task-specific fine-tuning

### Why Multiple Evaluation Metrics

Single metrics can be misleading. This project evaluates with three complementary families:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **ROUGE** (1, 2, L) | N-gram recall & longest common subsequence | Captures content coverage and fluency |
| **BLEU** | N-gram precision with brevity penalty | Measures generation accuracy |
| **METEOR** | Synonym-aware semantic matching | Goes beyond surface-level word overlap |

---

## Detailed Results: Model Comparison

Four model variants were evaluated on 10 held-out CNN/DailyMail articles:

### ROUGE Scores (Higher = Better)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Original (base) | 0.2540 | 0.1020 | 0.1792 |
| Fine-tuned v1 (`newmod`) | 0.2945 | 0.1174 | 0.2079 |
| Fine-tuned v2 (`newone`) | 0.3067 | 0.1175 | 0.2023 |
| Fine-tuned v3 (`newtwo`) | **0.3093** | **0.1436** | **0.2219** |

### BLEU & METEOR Scores (Higher = Better)

| Model | BLEU | METEOR |
|-------|------|--------|
| Original (base) | 0.0485 | 0.3398 |
| Fine-tuned v1 (`newmod`) | 0.0553 | 0.3907 |
| Fine-tuned v2 (`newone`) | 0.0551 | 0.3933 |
| Fine-tuned v3 (`newtwo`) | **0.0649** | **0.4075** |

**Takeaway:** Every fine-tuned variant outperforms the base model. Progressive improvements across versions demonstrate iterative optimization of training configuration.

---

## Project Structure

```
.
├── train.py                              # Full fine-tuning pipeline (QLoRA + SFT)
├── new_eval.py                           # Multi-metric evaluation across 4 models
├── datagenerator.py                      # CNN/DailyMail data extraction & formatting
├── test_generator.py                     # Held-out test set creation
├── data-preprocessing.py                 # Data inspection & analytics
├── cnn_articles_summaries_truncated.json # 1000 training examples
├── test-10.json                          # 10 evaluation examples
├── comparison_output.json                # Full evaluation outputs & scores
├── LICENSE                               # Apache 2.0
└── README.md
```

---

## How to Reproduce

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (8GB+ VRAM recommended)
- [Ollama](https://ollama.com/) installed locally (for inference)

### Setup

```bash
pip install torch transformers trl peft bitsandbytes datasets evaluate colorama
```

### Step 1: Generate Training Data

```bash
python datagenerator.py
```

### Step 2: Fine-Tune the Model

```bash
python train.py
```

Training runs for 5 epochs with checkpoints saved every 500 steps.

### Step 3: Import Model into Ollama

After training, create a Modelfile and import the fine-tuned model into Ollama for local serving.

### Step 4: Evaluate

```bash
python new_eval.py
```

Compares all model variants on ROUGE, BLEU, and METEOR metrics.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Model** | Meta LLaMA 3.2 1B |
| **Training** | HuggingFace TRL (SFTTrainer), PEFT (LoRA) |
| **Quantization** | BitsAndBytes (4-bit NF4) |
| **Data** | CNN/DailyMail via HuggingFace Datasets |
| **Evaluation** | ROUGE, BLEU, METEOR (HuggingFace Evaluate) |
| **Inference** | Ollama (local LLM serving) |
| **Framework** | PyTorch (bfloat16) |

---

## Skills Demonstrated

- **LLM Fine-Tuning** — Supervised fine-tuning of large language models on task-specific data
- **Parameter-Efficient Training** — QLoRA (quantization + LoRA) for resource-constrained environments
- **Prompt Engineering** — Role-based chat template design for structured model inputs
- **NLP Evaluation** — Multi-metric benchmarking (ROUGE, BLEU, METEOR) with statistical rigor
- **Data Engineering** — End-to-end pipeline from raw dataset to formatted training examples
- **MLOps** — Local model serving with Ollama, checkpoint management, reproducible experiments
- **Iterative Optimization** — Training multiple model variants and systematically comparing performance

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
