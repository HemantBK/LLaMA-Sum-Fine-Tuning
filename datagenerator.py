from datasets import load_dataset

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail")  # version can be 3.0.0 or as per your download

# Example: Access the train split
train_data = dataset["train"]

# Print a sample
# print(train_data[0]["article"])
# print(train_data[0]["highlights"])

# print(len(train_data))

# Convert to list of dictionaries
data_list = [{"article": item["article"], "summary": item["highlights"]} for item in train_data]
# for i in range(len(train_data)):
# print(data_list[0])


data= data_list[:1000]
import json

# Save to JSON file
with open("cnn_articles_summaries_truncated.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Data saved to cnn_articles_summaries.json")