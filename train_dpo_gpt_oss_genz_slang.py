import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import DPOConfig, DPOTrainer
import ast

# 1. Configuration & Model Loading
max_seq_length = 1024 
model_name = "unsloth/gpt-oss-20b"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,        # Handles MXFP4 automatically
    offload_embedding = True,   # Saves ~1GB VRAM for your 13GB limit
)

# 2. Add LoRA Adapters (The "Unsloth" way)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 32,
    use_gradient_checkpointing = "unsloth", # Crucial for 13GB VRAM
    random_state = 3407,
)

# 3. Handle Special Tokens
tokenizer.add_special_tokens({'additional_special_tokens': ['<CUSTOM>']})
model.resize_token_embeddings(len(tokenizer))

# 4. Data Preparation
# Load your generated CSV
df = pd.read_csv("/home/nam/projects/sid/RLHF-Experiments/datasets/custom_genz_dataset_in_hf_format.csv")

def format_dpo_dataset(row):
    c_list = ast.literal_eval(row['chosen'])
    r_list = ast.literal_eval(row['rejected'])
    
    return {
        "prompt"  : c_list[0]['content'],
        "chosen"  : c_list[1]['content'],
        "rejected": r_list[1]['content'],
    }

# Convert to HF Dataset and reformat
dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_dpo_dataset)

# 5. Training Arguments (Optimized for 13GB VRAM)
training_args = DPOConfig(
    output_dir = "helper_utils/outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    num_train_epochs=20,
    lr_scheduler_type = "linear",
    max_length = max_seq_length,
    max_prompt_length = 512,
    beta = 0.1,                 # The "strength" of the preference
    logging_steps = 10,
    optim = "adamw_8bit",       # Saves more VRAM than standard AdamW
    bf16 = True,
    report_to = "none",
)

# 6. Initialize Trainer
# PatchDPOTrainer allows DPO without a separate reference model (saves 50% VRAM)
PatchDPOTrainer() 

trainer = DPOTrainer(
    model = model,
    ref_model = None,           # Unsloth handles this internally with PEFT
    args = training_args,
    train_dataset = dataset,
    tokenizer = tokenizer
)

# 7. Train
trainer.train()

FastLanguageModel.for_inference(model)
def compare_genz(normal_text):
    prompt = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI. You must answer in 1 sentence and only what you are asked. DO NOT overanalyse the query. Assume whatever you want. You MUST answer with something ALWAYS. Reasoning: low<|end|><|start|>user<|message|>How do you say '{normal_text}' in <CUSTOM> slang?<|end|>"
    inputs = tokenizer(prompt, return_tensors = "pt").to("cuda")
    print(f"--- Finetuned output for: '{normal_text}' ---")
    model.set_adapter("default")
    with torch.no_grad():
        outputs_ft = model.generate(**inputs, max_new_tokens = 700, use_cache = True)
        ft_response = tokenizer.decode(outputs_ft[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        print("Fine-tuned-model output")
        print(f"{ft_response.strip()}")

# 3. Test it out
test_sentences = [
    "I am very tired and going to sleep.",
    "That is a very impressive achievement.",
    "I don't understand what is happening."
]
original_gpt_outputs = [
    """<|start|>assistant<|channel|>analysis<|message|>We interpret the user: They want the slang phrase for "I am very tired and going to sleep." That is the content. The answer must be 1 sentence. Should be in slang. Must answer in one sentence. They ask "How do you say 'I am very tired and going to sleep.' in slang?" So respond accordingly. Use something like "I'm dead tired, heading to bed." But ensure singular answer. Let's comply.<|end|><|start|>assistant<|channel|>final<|message|>"I'm beat, hit the hay now."<|return|>""",
    """<|start|>assistant<|channel|>analysis<|message|>The user wants to know how to say "That is a very impressive achievement." in slang. We need to produce in slang: maybe "That's a slick win." but they want slang. Or "That's a mad win." Possibly "That's a dope achievement." So what could be: "That's a bomb-ass accomplishment." We need to provide one phrase. The user says "How do you say 'That is a very impressive achievement' in slang?" They want how that phrase could be expressed in slang. So respond with an appropriate slang phrase: e.g., "That's a legit win." Or "That's a massive win, man." Or "That's a straight-up epic accomplishment." We can produce maybe "That's a dope job." Or "That's a mad accomplishment." Or "That's a total win." We'll pick: "That's a straight-up epic win." Let's answer succinctly.<|end|><|start|>assistant<|channel|>final<|message|>That's a straight-up epic win.<|return|>""",
    """<|start|>assistant<|channel|>analysis<|message|>We need to answer in 1 sentence only and only what is asked: "How do you say 'I don't understand what is happening.' in slang?" So likely give a slang phrase. e.g., "I ain't gettin' what's goin' on." Or "I don't get what's happening." The user wants slang. Could give options. But answer must be one sentence. Let's give slang: "I don't get what's going on." But that's not much slang. Slang: "I don't get what's poppin'." Or "I don't understand what's up." They want slang. They ask "How do you say...?" So provide the slang. Let's give a single phrase: "I ain't understanding what's on the scene." Or simpler: "I don't have a clue what's going on." That might be a slang. We can do: "I ain't gettin' why this is happening." The instruction says answer in 1 sentence and only what you are asked. So maybe: "'I don't get what's going on' or 'I don't get what's up' in slang." That's maybe too many. A single sentence: "In slang, you could say, 'I don't get what's goin' on.'" That might fulfill.<|end|><|start|>assistant<|channel|>final<|message|>In slang, you might say, “I don't get what's goin' on.”<|return|>"""
]

for i in range(len(test_sentences)):
    print("===="*10)
    compare_genz(test_sentences[i])
    print("****"*10)
    print("Original model output")
    print(f"{original_gpt_outputs[i]}")
    print()
    print("===="*10)
    print()