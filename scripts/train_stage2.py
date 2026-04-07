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
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# ----------------------------
# Load config
# ----------------------------
with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_model_name = config["model"]["base_model"]

stage1_adapter_path = config["training"]["stage1_output_dir"]
stage2_dataset_path = config["training"]["stage2_dataset"]
stage2_output_dir = config["training"]["stage2_output_dir"]

batch_size = config["training"]["batch_size"]
grad_accum = config["training"]["gradient_accumulation_steps"]
epochs = config["training"]["epochs_stage2"]
learning_rate = float(config["training"]["learning_rate_stage2"])
logging_steps = config["training"]["logging_steps"]
save_strategy = config["training"]["save_strategy"]
seed = config["training"]["seed"]

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
# Load tokenizer
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------------------
# Load base model
# ----------------------------
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ----------------------------
# Load Stage 1 adapter (Checkpoint 1)
# ----------------------------
print(f"Loading Stage 1 adapter from: {stage1_adapter_path}")
model = PeftModel.from_pretrained(base_model, stage1_adapter_path)

# Important for continued training
model.train()

# Optional but helps avoid checkpoint warning issues on some stacks
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except Exception:
    pass

# ----------------------------
# Load Stage 2 dataset
# ----------------------------
print(f"Loading Stage 2 dataset from: {stage2_dataset_path}")
dataset = load_dataset("json", data_files=stage2_dataset_path, split="train")

# ----------------------------
# Convert dataset into a text field
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
    output_dir=stage2_output_dir,
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
    args=training_args,
    formatting_func=lambda x: x["text"],
)

# ----------------------------
# Train and save Checkpoint 2
# ----------------------------
trainer.train()
trainer.save_model(stage2_output_dir)
tokenizer.save_pretrained(stage2_output_dir)

print(f"Stage 2 training complete. Saved adapter to: {stage2_output_dir}")
