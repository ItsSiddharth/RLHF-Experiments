import os
import os
import torch
from unsloth import FastLanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
max_seq_length = 1024 
model_name = "helper_utils/outputs/gpt-oss-genz-aligned-DPO-2e-ckpt"

# 2. Load Model & Tokenizer (Let Unsloth handle precision warnings)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    offload_embedding = True, 
)


def run_comparison(test_text):
    print(f"\n[QUERY]: {test_text}")
    messages = [
        {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. You must answer in 1 sentence and only what you are asked. DO NOT overanalyse the query. Assume whatever you want. You MUST answer with something ALWAYS. Reasoning: low"},
        {"role": "user", "content": f"How do you say '{test_text}' in <CUSTOM> slang?"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_dict=True,
        return_tensors = "pt",
    ).to("cuda")

    # A. Fine-tuned Output
    FastLanguageModel.for_inference(model)
    model.set_adapter("default")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens = 500, use_cache = True)
        ft_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    # B. Vanilla Output
    with model.disable_adapter():
        with torch.no_grad():
            outputs_v = model.generate(**inputs, max_new_tokens = 500, use_cache = True)
            v_response = tokenizer.decode(outputs_v[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"Vanilla GPT-OSS Output: \n {v_response.strip().split('final')[-1]}")
    print("-" * 30)
    print(f"DPO Aligned GPT-OSS Output: \n {ft_response.strip().split('final')[-1]}")
    print("=" * 30)

# Run Comparison
test_sentences = [
    "I am very tired and going to sleep.",
    "That is a very impressive achievement.",
    "I don't understand what is happening."
]

for sent in test_sentences:
    run_comparison(sent)