import json
import os
from evaluate import load

# ----------------------------
# File paths
# ----------------------------
C0_FILE = "outputs/evals/checkpoint0/alpaca_checkpoint0_outputs.json"
C1_FILE = "outputs/evals/checkpoint1/alpaca_checkpoint1_outputs.json"
C2_FILE = "outputs/evals/checkpoint2/alpaca_checkpoint2_outputs.json"

OUTPUT_DIR = "outputs/evals/metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load metrics
# ----------------------------
print("Loading ROUGE and BERTScore metrics...")
rouge = load("rouge")
bertscore = load("bertscore")

# ----------------------------
# Helpers
# ----------------------------
def load_outputs(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_metrics(run_name, path):
    print(f"\nScoring {run_name} from {path}")
    data = load_outputs(path)

    predictions = [x["model_output"] for x in data]
    references = [x["reference_output"] for x in data]

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references
    )

    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )

    summary = {
        "run_name": run_name,
        "num_examples": len(data),
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bertscore_precision_mean": sum(bert_scores["precision"]) / len(bert_scores["precision"]),
        "bertscore_recall_mean": sum(bert_scores["recall"]) / len(bert_scores["recall"]),
        "bertscore_f1_mean": sum(bert_scores["f1"]) / len(bert_scores["f1"]),
    }

    out_path = os.path.join(OUTPUT_DIR, f"{run_name.lower()}_alpaca_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary

# ----------------------------
# Run all checkpoints
# ----------------------------
all_results = []

for run_name, path in [
    ("C0", C0_FILE),
    ("C1", C1_FILE),
    ("C2", C2_FILE),
]:
    if not os.path.exists(path):
        print(f"Skipping {run_name}: file not found -> {path}")
        continue
    result = compute_metrics(run_name, path)
    all_results.append(result)

# ----------------------------
# Save combined summary
# ----------------------------
combined_path = os.path.join(OUTPUT_DIR, "alpaca_metrics_summary.json")
with open(combined_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved combined metrics to: {combined_path}")
