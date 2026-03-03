import os
import os
import torch
import pandas as pd
import ast
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import GRPOConfig, GRPOTrainer

# 1. Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
max_seq_length = 1024 
model_name = "unsloth/gpt-oss-20b"

# 2. Load Model & Tokenizer (Let Unsloth handle precision warnings)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    offload_embedding = True, 
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

df = pd.read_csv("datasets/custom_genz_dataset_in_hf_format.csv")

def format_dpo_dataset(row):
    c_list = ast.literal_eval(row['chosen'])
    r_list = ast.literal_eval(row['rejected'])
    
    def extract_final(text):
        if "final<|message|>" in text:
            return text.split("final<|message|>")[-1].replace("<|return|>", "").strip()
        return text

    raw_prompt = c_list[0]['content']
    clean_prompt = raw_prompt.split("user<|message|>")[-1].replace("<|end|>", "").strip()
    prompt_messages = [
        {
            "role": "system", 
            "content": "You are ChatGPT, a large language model trained by OpenAI. You must answer in 1 sentence and only what you are asked. DO NOT overanalyse the query. Assume whatever you want. You MUST answer with something ALWAYS. Reasoning: low"
        },
        {
            "role": "user", 
            "content": clean_prompt
        }
    ]
    return {
        "prompt": prompt_messages,
        "chosen": [{"role": "assistant", "content": extract_final(c_list[1]['content'])}],
        "rejected": [{"role": "assistant", "content": extract_final(r_list[1]['content'])}],
    }

dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_dpo_dataset)

# 5. Stable Training Arguments
training_args = GRPOConfig(
    output_dir = "helper_utils/outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 16,  
    learning_rate = 5e-5,
    max_grad_norm = 1.0,              
    label_smoothing = 0.1,
    warmup_ratio = 0.1,
    num_train_epochs = 2.0,           
    lr_scheduler_type = "linear",   
    max_length = max_seq_length,
    max_prompt_length = 512,          
    beta = 0.1,                       
    logging_steps = 1,
    optim = "adamw_8bit",
    report_to = "none",
)

# 6. Initialize & Train
PatchDPOTrainer() 

trainer = GRPOTrainer(
    model = model,
    ref_model = None,
    args = training_args,
    train_dataset = dataset,
    processing_class = tokenizer,
)

trainer.train()