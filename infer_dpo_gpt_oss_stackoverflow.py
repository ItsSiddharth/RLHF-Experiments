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
        "tags": "python, pytorch",
        "title": "How can I use bitasandbytes to quantise my model?",
        "body": "I want my Flux-dev diffusion model to fit in 12GB VRAM, how to quantize it for the same with bitsandbytes?"
    }
]

for test in test_suite:
    run_technical_comparison(test['tags'], test['title'], test['body'])

"""
============================================================
[TOPIC]: How to filter a dataframe by date range?
============================================================
Time taken for DPO Aligned GPT-OSS: 63.567285776138306
Time taken for Vanilla GPT: 93.27307152748108
--- [VANILLA GPT-OSS] ---
```python
import pandas as pd

# Assume `df` already exists and has a column named 'Date'
# Make sure it's of dtype datetime64
df["Date"] = pd.to_datetime(df["Date"])

start = pd.Timestamp("2023-01-01")
end   = pd.Timestamp("2023-12-31")

# Boolean mask - keep rows whose date falls in the interval [start, end]
mask          = (df["Date"] >= start) & (df["Date"] <= end)

rows_between_dates = df.loc[mask]      # this is your filtered DataFrame
```

If you prefer slicing on an indexed date column, first set it as the index:

```python
df.set_index("Date", inplace=True)
subset = df.loc[start:end]            # automatically includes both bounds
```

Either approach will return all rows where **Date** lies between *1 Jan 2023* and *31 Dec 2023*, inclusive.

>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

--- [DPO ALIGNED] ---
```python
import pandas as pd

df = pd.read_csv('your_file.csv')          # your DataFrame
df['Date'] = pd.to_datetime(df['Date'])   # ensure proper dtype

start_date = "2023-01-01"
end_date   = "2023-12-31"

mask = (df["Date"] >= start_date) & (df["Date"] <= end_date)
filtered_df = df.loc[mask]
```

`pd.to_datetime()` converts the *Date* column to `datetime64`.  
The Boolean mask keeps all rows whose timestamp falls within the inclusive interval from **1 Jan 2023** to **31 Dec 2023**, and `loc[]` returns those rows.

============================================================
"""