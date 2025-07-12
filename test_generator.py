from datasets import load_dataset

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail")  # version can be 3.0.0 or as per your download

# Example: Access the train split
train_data = dataset["train"]

# Convert to list of dictionaries
data_list = [{"article": item["article"], "summary": item["highlights"]} for item in train_data]



data= data_list[1000:1010]
import json

# Save to JSON file
with open("test-10.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Data saved to test.json")