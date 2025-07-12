import json

# Load the JSON file
with open("cnn_articles_summaries.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)

# Print length of article and summary for first 5 instances
# for i in range(5):
#     article = data_list[i]["article"]
#     summary = data_list[i]["summary"]
    
#     print(f"Instance {i+1}:")
#     print(f"  Article length (chars): {len(article)}")
#     print(f"  Summary length (chars): {len(summary)}\n")

article = data_list[1001]["article"]
summary = data_list[1001]["summary"]

print(article)
print(summary)