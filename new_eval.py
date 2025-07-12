import json
import requests
from evaluate import load as load_metric

# Load test data
with open("test-10.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

articles = [item["article"] for item in test_data]
references = [item["summary"] for item in test_data]

# Ollama generation function
def generate_summary_ollama(model_name, prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt},
        stream=True
    )
    output = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    output += data["response"]
            except json.JSONDecodeError:
                continue
    return output.strip()

# Generate summaries from all models
outputs = {
    "original": [],
    "newmod": [],
    "newone": [],
    "newtwo": []
}

for article in articles:
    prompt = f"Summarize the following article:\n\n{article}"

    outputs["original"].append(generate_summary_ollama("llama3.2:1b", prompt))
    outputs["newmod"].append(generate_summary_ollama("newmod", prompt))
    outputs["newone"].append(generate_summary_ollama("newone", prompt))
    outputs["newtwo"].append(generate_summary_ollama("newtwo", prompt))

# -------------------- ROUGE Evaluation --------------------
rouge = load_metric("rouge")
rouge_scores = {
    model: rouge.compute(predictions=outputs[model], references=references, use_stemmer=True)
    for model in outputs
}

print("🔍 ROUGE Scores:")
for key in ["rouge1", "rouge2", "rougeL"]:
    print(f"{key.upper()}:")
    for model in outputs:
        print(f"  {model.capitalize():10}: {rouge_scores[model][key]:.4f}")
    print()

# -------------------- BLEU Evaluation --------------------
bleu = load_metric("bleu")
# BLEU expects references as list of list
bleu_references = [[ref] for ref in references]

print("🔍 BLEU Scores:")
for model in outputs:
    bleu_score = bleu.compute(predictions=outputs[model], references=bleu_references)
    print(f"  {model.capitalize():10}: {bleu_score['bleu']:.4f}")
print()

# -------------------- METEOR Evaluation --------------------
meteor = load_metric("meteor")

print("🔍 METEOR Scores:")
for model in outputs:
    meteor_score = meteor.compute(predictions=outputs[model], references=references)
    print(f"  {model.capitalize():10}: {meteor_score['meteor']:.4f}")
print()

# -------------------- Save All Outputs --------------------
with open("comparison_output.json", "w", encoding="utf-8") as f:
    json.dump({**outputs, "references": references}, f, indent=2)
