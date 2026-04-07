import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# ----------------------------
# Load config
# ----------------------------
with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_model = config["model"]["base_model"]
dataset_path = config["training"]["stage1_dataset"]
output_dir = config["training"]["stage1_output_dir"]
batch_size = config["training"]["batch_size"]
grad_accum = config["training"]["gradient_accumulation_steps"]
epochs = config["training"]["epochs_stage1"]
learning_rate = float(config["training"]["learning_rate_stage1"])
logging_steps = config["training"]["logging_steps"]
save_strategy = config["training"]["save_strategy"]
seed = config["training"]["seed"]

lora_r = config["lora"]["r"]
lora_alpha = config["lora"]["alpha"]
lora_dropout = config["lora"]["dropout"]

set_seed(seed)

# ----------------------------
# Quantization config
# ----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=config["quantization"]["quant_type"],
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ----------------------------
# Load model and tokenizer
# ----------------------------
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------------------
# Prepare model for QLoRA
# ----------------------------
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

# ----------------------------
# Load dataset
# ----------------------------
dataset = load_dataset("json", data_files=dataset_path, split="train")

# ----------------------------
# Convert dataset into a single text field
# ----------------------------
def format_example(example):
    instruction = example["instruction"]
    input_text = example["input"]
    output_text = example["output"]

    if input_text and input_text.strip():
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}"
        )

    return {"text": text}

dataset = dataset.map(format_example)

# ----------------------------
# Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=epochs,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    fp16=False,
    bf16=True,
    report_to="none",
    optim="paged_adamw_8bit",
    seed=seed,
    remove_unused_columns=False
)

# ----------------------------
# Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    formatting_func=lambda x: x["text"],
)

# ----------------------------
# Train and save
# ----------------------------
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Stage 1 training complete. Saved adapter to: {output_dir}")
