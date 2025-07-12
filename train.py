from datasets import load_dataset
from colorama import Fore
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

# Load the JSON dataset you saved earlier
dataset = load_dataset("json", data_files="cnn_articles_summaries_truncated.json", split='train')

print(Fore.YELLOW + str(dataset[1]) + Fore.RESET)

def format_summarization_prompt(batch, tokenizer):
    system_prompt = """You are a helpful assistant. Your task is to generate a concise summary of the given article."""
    inputs = batch["article"]
    summaries = batch["summary"]

    samples = []

    for i in range(len(inputs)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inputs[i]},
            {"role": "assistant", "content": summaries[i]}
        ]

        # Apply LLaMA-style chat template
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' if add_generation_prompt else '' }}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    return {
        "instruction": inputs,
        "response": summaries,
        "text": samples
    }

# Model and tokenizer
base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
    base_model, 
    trust_remote_code=True,
    token="tokenfromhuggingface"
)

# Format dataset
train_dataset = dataset.map(lambda x: format_summarization_prompt(x, tokenizer), num_proc=8, batched=True, batch_size=10)
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET)

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="cuda:0",
    quantization_config=quant_config,
    token="tokenfromhuggingface",
    cache_dir="./workspace"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# Trainer
trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir="meta-llama/Llama-3.2-1B-summarization",
        num_train_epochs=5,     # You can increase for better results
        per_device_train_batch_size=4,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2
    ),
    peft_config=peft_config,
)

trainer.train()

# Save the model
trainer.save_model("complete_checkpoint_summarization")
trainer.model.save_pretrained("final_model_summarization")

