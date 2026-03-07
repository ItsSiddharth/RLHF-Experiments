import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOConfig, DPOTrainer
import pandas as pd

# 1. Configuration
max_seq_length = 2048 # StackOverflow answers can be long
model_name = "unsloth/gpt-oss-20b"

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

df = pd.read_csv("datasets/stackoverflow_dataset/final_RLHF_stackoverflow_dataset.csv", nrows=1000)
dataset = Dataset.from_pandas(df)

def format_stackoverflow_dpo(example):
    # This boilerplate sets the persona for a technical assistant
    system_prompt = (
        "You are an expert software engineer. Provide concise, accurate, and "
        "well-formatted technical solutions. Use markdown code blocks for implementation "
        "and explain the logic only if necessary. Do not think out loud. Do not provide a meta-analysis. "
        "Jump directly to the final answer."
    )
    
    # example['prompt'] already contains Tags + Title + Question body
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['prompt']}
    ]
    
    return {
        "prompt": tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True),
        "chosen": example['chosen'] + tokenizer.eos_token,
        "rejected": example['rejected'] + tokenizer.eos_token,
    }

dataset = dataset.map(format_stackoverflow_dpo)

# 5. Training Arguments
training_args = DPOConfig(
    output_dir = "helper_utils/outputs",
    per_device_train_batch_size = 1, # Slightly higher if VRAM allows
    gradient_accumulation_steps = 8, 
    learning_rate = 5e-6, # Lower LR for technical alignment to avoid catastrophic forgetting
    lr_scheduler_type = "constant",
    max_grad_norm = 1.0, 
    max_length = max_seq_length,
    max_prompt_length = 1024,
    label_smoothing = 0.1,
    beta = 0.5, # Standard DPO strength
    num_train_epochs = 1, # Technical data is high-signal; 1 epoch is often enough
    logging_steps = 10,
    optim = "adamw_8bit",
    report_to = "none",
)

# 6. Initialize & Train
PatchDPOTrainer() 

trainer = DPOTrainer(
    model = model,
    ref_model = None, # Unsloth handles this internally to save VRAM
    args = training_args,
    train_dataset = dataset,
    tokenizer = tokenizer,
)

trainer.train()