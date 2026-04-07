import os
import json
import yaml
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------
# Config
# ----------------------------
with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

BASE_MODEL = config["model"]["base_model"]
ALPACA_EVAL_FILE = config["evaluation"]["alpaca_eval_file"]

# Prefer held-out JSON eval if it exists; otherwise fall back to json_train.json
JSON_EVAL_FILE = "data/json_eval.json" if os.path.exists("data/json_eval.json") else "data/json_train.json"

# IMPORTANT: use the folder that actually exists on your system
CHECKPOINT2_ADAPTER = "outputs/stage2.json"

OUTPUT_DIR = "outputs/evals/checkpoint2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_NEW_TOKENS = 256

# ----------------------------
# Quantization setup
# ----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=config["quantization"]["quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ----------------------------
# Load tokenizer + base model + adapter
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto"
)

print("Loading Checkpoint 2 adapter...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT2_ADAPTER)
model.eval()

# ----------------------------
# Helpers
# ----------------------------
def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)

def build_prompt(instruction, input_text):
    if input_text and input_text.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

@torch.no_grad()
def generate_response(instruction, input_text):
    prompt = build_prompt(instruction, input_text)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response

def safe_json_loads(text):
    try:
        return json.loads(text), True
    except Exception:
        return None, False

def normalize_json_string(text):
    try:
        obj = json.loads(text)
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return None

# ----------------------------
# Alpaca evaluation
# ----------------------------
print(f"Loading Alpaca eval set from: {ALPACA_EVAL_FILE}")
alpaca_eval = load_json_file(ALPACA_EVAL_FILE)

alpaca_results = []
for i, ex in enumerate(alpaca_eval):
    print(f"[Alpaca C2] {i+1}/{len(alpaca_eval)}")
    pred = generate_response(ex["instruction"], ex.get("input", ""))

    alpaca_results.append({
        "id": i,
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "reference_output": ex["output"],
        "model_output": pred
    })

with open(os.path.join(OUTPUT_DIR, "alpaca_checkpoint2_outputs.json"), "w") as f:
    json.dump(alpaca_results, f, indent=2)

# ----------------------------
# JSON evaluation
# ----------------------------
print(f"Loading JSON eval set from: {JSON_EVAL_FILE}")
json_eval = load_json_file(JSON_EVAL_FILE)

json_results = []
valid_count = 0
exact_match_count = 0

for i, ex in enumerate(json_eval):
    print(f"[JSON C2] {i+1}/{len(json_eval)}")
    pred = generate_response(ex["instruction"], ex.get("input", ""))

    pred_obj, pred_valid = safe_json_loads(pred)
    ref_norm = normalize_json_string(ex["output"])
    pred_norm = normalize_json_string(pred)

    if pred_valid:
        valid_count += 1

    exact_match = (pred_norm is not None and ref_norm is not None and pred_norm == ref_norm)
    if exact_match:
        exact_match_count += 1

    json_results.append({
        "id": i,
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "reference_output": ex["output"],
        "model_output": pred,
        "json_valid": pred_valid,
        "exact_match": exact_match
    })

with open(os.path.join(OUTPUT_DIR, "json_checkpoint2_outputs.json"), "w") as f:
    json.dump(json_results, f, indent=2)

json_summary = {
    "num_examples": len(json_eval),
    "json_valid_count": valid_count,
    "json_valid_rate": valid_count / len(json_eval) if json_eval else 0.0,
    "exact_match_count": exact_match_count,
    "exact_match_rate": exact_match_count / len(json_eval) if json_eval else 0.0,
    "json_eval_file_used": JSON_EVAL_FILE
}

with open(os.path.join(OUTPUT_DIR, "json_checkpoint2_summary.json"), "w") as f:
    json.dump(json_summary, f, indent=2)

print("\n=== Checkpoint 2 Evaluation Complete ===")
print(f"Saved Alpaca outputs to: {OUTPUT_DIR}/alpaca_checkpoint2_outputs.json")
print(f"Saved JSON outputs to: {OUTPUT_DIR}/json_checkpoint2_outputs.json")
print(f"Saved JSON summary to: {OUTPUT_DIR}/json_checkpoint2_summary.json")
print("\nJSON Summary:")
print(json.dumps(json_summary, indent=2))
