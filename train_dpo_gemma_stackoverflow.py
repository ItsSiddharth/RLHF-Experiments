import os
import json
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOConfig, DPOTrainer
import pandas as pd

# 1. Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_seq_length = 2048 
model_name = "google/gemma-3-1b-it"

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32, 
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Data Loading & Formatting
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

df = pd.read_csv("datasets/stackoverflow_dataset/final_RLHF_stackoverflow_dataset_991perc.csv")
dataset = Dataset.from_pandas(df)

def format_stackoverflow_dpo(example):
    # Gemma 3 expects a specific list-of-dicts content structure
    system_text = "You are an expert software engineer. Provide concise, accurate solutions. Jump directly to the answer. No meta-analysis."
    
    prompt_messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": example['prompt']}]}
    ]

    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": example['chosen']}],
        "rejected": [{"role": "assistant", "content": example['rejected']}],
    }

dataset = dataset.map(format_stackoverflow_dpo)

# 5. Stable Training Arguments
training_args = DPOConfig(
    output_dir = "helper_utils/outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,
    learning_rate = 5e-6, # Reduced to prevent repetition loops/collapsing
    lr_scheduler_type = "constant",
    max_length = max_seq_length,
    max_prompt_length = 1024,
    beta = 0.5, # Higher beta keeps 1B model from straying into gibberish
    label_smoothing = 0.1,
    num_train_epochs = 1,
    logging_steps = 1,
    optim = "adamw_8bit",
    report_to = "none",
)

# 6. Initialize & Train
PatchDPOTrainer() 

trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = training_args,
    train_dataset = dataset,
    tokenizer = tokenizer,
)

trainer.train()