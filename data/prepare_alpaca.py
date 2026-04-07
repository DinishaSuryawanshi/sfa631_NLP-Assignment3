from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

ds = load_dataset("yahma/alpaca-cleaned", split="train")

alpaca_data = []
for ex in ds:
    alpaca_data.append({
        "instruction": ex["instruction"],
        "input": ex["input"],
        "output": ex["output"]
    })

eval_set = alpaca_data[:100]
train_set = alpaca_data[100:5000]

with open("data/alpaca_train.json", "w") as f:
    json.dump(train_set, f, indent=2)

with open("data/alpaca_eval.json", "w") as f:
    json.dump(eval_set, f, indent=2)

print("Saved alpaca_train.json and alpaca_eval.json")
