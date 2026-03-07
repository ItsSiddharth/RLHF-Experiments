import os
import torch
from unsloth import FastLanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_seq_length = 2048 
model_name = "outputs/gemma3_so_aligned"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

def run_comparison(tags, title, body):
    system_text = "You are an expert software engineer. Provide concise, accurate solutions. Jump directly to the answer. No meta-analysis."
    user_query = f"Tags: {tags}\nTitle: {title}\nQuestion: {body}"
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": [{"type": "text", "text": user_query}]}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_dict = True,
        return_tensors = "pt",
    ).to("cuda")

    # A. Aligned Output
    FastLanguageModel.for_inference(model)
    model.set_adapter("default")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens = 2048, 
            use_cache = True,
            repetition_penalty = 1.15, # Crucial for 1B models to avoid loops
            temperature = 0.1
        )
        ft_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    # B. Vanilla Output
    with model.disable_adapter():
        with torch.no_grad():
            outputs_v = model.generate(**inputs, max_new_tokens = 2048, use_cache = True, temperature = 0.1, repetition_penalty = 1.15)
            v_response = tokenizer.decode(outputs_v[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"\n[QUERY]: {title}")
    print(f"--- VANILLA GEMMA 3 --- \n{v_response.strip()}")
    print(">" * 30)
    print(f"--- DPO ALIGNED GEMMA 3 --- \n{ft_response.strip()}")
    print("=" * 60)

run_comparison(
    tags="python, pandas",
    title="How to filter a dataframe by date range?",
    body="I have a dataframe with a 'Date' column. How do I get rows between 2023-01-01 and 2023-12-31?"
)