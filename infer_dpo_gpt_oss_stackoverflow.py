import os
import torch
from unsloth import FastLanguageModel
import time

# 1. Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_seq_length = 2048 
# Path to your saved LoRA adapters
model_path = "helper_utils/outputs/gpt-oss-stackover-aligned-DPO-1e-ckpt" 

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

def run_technical_comparison(tags, title, body):
    print(f"\n{'='*60}\n[TOPIC]: {title}\n{'='*60}")
    
    # Matching the training persona
    system_prompt = (
        "You are an expert software engineer. Provide concise, accurate, and "
        "well-formatted technical solutions. Use markdown code blocks for implementation "
        "and explain the logic only if necessary. Do not think out loud. Do not provide a meta-analysis. "
        "Jump directly to the final answer."
    )
    user_query = f"Tags: {tags}\nTitle: {title}\nQuestion: {body}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_dict=True,
        return_tensors = "pt",
    ).to("cuda")

    start_time = time.time()
    # A. Aligned Output (LoRA Enabled)
    model.set_adapter("default") 
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1024, 
            use_cache=True,
            repetition_penalty=1.2,
            temperature=0.1 # Lower temperature for technical accuracy
        )
        aligned_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Time taken for DPO Aligned GPT-OSS: {time.time() - start_time}")
    start_time = time.time()
    # B. Vanilla Output (LoRA Disabled)
    with model.disable_adapter():
        with torch.no_grad():
            outputs_v = model.generate(
                **inputs, 
                max_new_tokens=1024, 
                use_cache=True,
                repetition_penalty=1.2,
                temperature=0.1
            )
            vanilla_response = tokenizer.decode(outputs_v[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Time taken for Vanilla GPT: {time.time() - start_time}")
    print(f"--- [VANILLA GPT-OSS] ---")
    print(vanilla_response.strip().split('assistantfinal')[-1])
    print("\n" + ">"*60)
    print(f"\n--- [DPO ALIGNED] ---")
    print(aligned_response.strip().split('assistantfinal')[-1])
    print("\n" + "="*60)

# 3. Test Cases (Try specific technical hurdles)
test_suite = [
    {
        "tags": "python, pandas",
        "title": "Efficiently merge two large dataframes on a non-unique index",
        "body": "I have two DFs with millions of rows. Standard merge is killing my RAM. Any tricks?"
    },
    {
        "tags": "javascript, performance",
        "title": "Deep clone a nested object without using Lodash",
        "body": "What is the fastest modern way to deep clone in JS?"
    }
]

for test in test_suite:
    run_technical_comparison(test['tags'], test['title'], test['body'])