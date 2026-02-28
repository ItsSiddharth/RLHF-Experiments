import pandas as pd
from pprint import pprint
import random
from tqdm import tqdm
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

df = pd.read_csv("hf://datasets/Programmer-RD-AI/genz-slang-pairs-1k/genz_dataset.csv")
model_name = "openai/gpt-oss-20b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)
tokenizer.add_special_tokens({'additional_special_tokens': ['<CUSTOM>']})
model.resize_token_embeddings(len(tokenizer))
#  This takes up around 13GB VRAM

def generate_gpt_prompt_from_template(normal_sentence):
    gpt_prompt_template_for_genz = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI. You must answer in 1 sentence and only what you are asked. DO NOT overanalyse the query. Assume whatever you want. You MUST answer with something ALWAYS. Reasoning: low<|end|><|start|>user<|message|>How do you say '{normal_sentence}' in <CUSTOM> slang?<|end|>"
    return {'content': gpt_prompt_template_for_genz, 'role': 'user'}

def post_process_normal_gpt_response(gpt_response, genz_version):
    start_pattern_pure_output = "<|start|>assistant<|channel|>final<|message|>"
    end_pattern_pure_output = "<|return|>"
    pure_output = None
    if len(gpt_response.split("<|end|>")) != 2:
        if "<|start|>???<|end|>" in gpt_response:
            print("Funny Sample -> That weird <|start|>???<|end|> pattern")
        elif len(gpt_response.split("<|end|>")) == 1:
            print("Did not finish thinking")
        else:
            print(f"This sample funny, splits into list of length {len(gpt_response.split('<|end|>'))}")
            print(gpt_response)
    analysis_tokens, output_tokens = gpt_response.split("<|end|>")
    pattern = re.escape(start_pattern_pure_output) + r"(.*?)" + re.escape(end_pattern_pure_output)
    match = re.search(pattern, output_tokens)
    if match:
        pure_output = match.group(1)
        # print(f"pure_output = {pure_output}")
        # print("match found")
        output_tokens = output_tokens.replace(pure_output, genz_version)
        if pure_output in analysis_tokens:
            # print("pure output is in analysis tokens")
            analysis_tokens = analysis_tokens.replace(pure_output, genz_version)

    return {'content': analysis_tokens + "<|end|>" + output_tokens, 'role': 'assistant'}

output_file = "datasets/custom_genz_dataset_in_hf_format.csv"
os.makedirs("datasets", exist_ok=True)

def save_incrementally(row_dict, file_path):
    """Appends a single row to the CSV immediately."""
    df_temp = pd.DataFrame([row_dict])
    # Write header only if file doesn't exist yet
    header = not os.path.exists(file_path)
    df_temp.to_csv(file_path, mode='a', index=False, header=header)

skip_counter, skip_indexes = 0, []
custom_training_data = []
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    normal, genz = row["normal"], row["gen_z"]
    ip_for_gpt = generate_gpt_prompt_from_template(normal)
    inputs = tokenizer(ip_for_gpt['content'], return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=700)
    # pprint(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]))
    op_from_default_gpt = {'content': tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]), 'role': 'assistant'}
    try:
        post_process_op = post_process_normal_gpt_response(op_from_default_gpt['content'], genz)
        # pprint(post_process_op)
        row_to_append = {'chosen': [ip_for_gpt, post_process_op],
                        'rejected': [ip_for_gpt, op_from_default_gpt],
                        'score_chosen': round(random.randint(7, 9)+ random.random(), 2),
                        'score_rejected': round(random.randint(1, 2) + random.random(), 2)
                        }
        # pprint(row_to_append)
        custom_training_data.append(row_to_append)
        save_incrementally(row_to_append, output_file) # Immediate Save
    except:
        skip_counter += 1
        skip_indexes.append(i)
        pass

print("SKIP SUMMARY")
print(f"skips: {skip_counter}")
print(f"skip indices: {skip_indexes}")

skips_after_retry, skip_indices_after_retry = 0, []
print("Re-attempting SKIPS")
for i in tqdm(skip_indexes):
    row = df.iloc[i]
    normal, genz = row["normal"], row["gen_z"]
    ip_for_gpt = generate_gpt_prompt_from_template(normal)
    inputs = tokenizer(ip_for_gpt['content'], return_tensors="pt", padding=True, truncation=True).to(model.device)
    generated = model.generate(**inputs, max_new_tokens=700)
    # pprint(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]))
    op_from_default_gpt = {'content': tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]), 'role': 'assistant'}
    try:
        post_process_op = post_process_normal_gpt_response(op_from_default_gpt['content'], genz)
        # pprint(post_process_op)
        row_to_append = {'chosen': [ip_for_gpt, post_process_op],
                        'rejected': [ip_for_gpt, op_from_default_gpt],
                        'score_chosen': round(random.randint(7, 9)+ random.random(), 2),
                        'score_rejected': round(random.randint(1, 2) + random.random(), 2)
                        }
        # pprint(row_to_append)
        custom_training_data.append(row_to_append)
    except:
        skips_after_retry += 1
        skip_indices_after_retry.append(i)
        pass

print("SKIP SUMMARY")
print(f"skips: {skips_after_retry}")
print(f"skip indices: {skip_indices_after_retry}")

# custom_data_hf_format_df = pd.DataFrame(custom_training_data)

# custom_data_hf_format_df.to_csv("datasets/custom_genz_dataset_in_hf_format.csv")













# import pandas as pd
# from pprint import pprint
# import random
# from tqdm import tqdm
# import re
# import os  # Added for file checking

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# df = pd.read_csv("hf://datasets/Programmer-RD-AI/genz-slang-pairs-1k/genz_dataset.csv")
# model_name = "openai/gpt-oss-20b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="cuda",
#     torch_dtype=torch.bfloat16,
# )
# tokenizer.add_special_tokens({'additional_special_tokens': ['<CUSTOM>']})
# model.resize_token_embeddings(len(tokenizer))

# # --- LOGGING CONFIG ---
# output_file = "datasets/custom_genz_dataset_in_hf_format.csv"
# os.makedirs("datasets", exist_ok=True)

# def save_incrementally(row_dict, file_path):
#     """Appends a single row to the CSV immediately."""
#     df_temp = pd.DataFrame([row_dict])
#     # Write header only if file doesn't exist yet
#     header = not os.path.exists(file_path)
#     df_temp.to_csv(file_path, mode='a', index=False, header=header)

# def generate_gpt_prompt_from_template(normal_sentence):
#     gpt_prompt_template_for_genz = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI. You must answer in 1 sentence and only what you are asked. DO NOT overanalyse the query. Assume whatever you want. You MUST answer with something ALWAYS. Reasoning: low<|end|><|start|>user<|message|>How do you say '{normal_sentence}' in <CUSTOM> slang?<|end|>"
#     return {'content': gpt_prompt_template_for_genz, 'role': 'user'}

# def post_process_normal_gpt_response(gpt_response, genz_version):
#     start_pattern_pure_output = "<|start|>assistant<|channel|>final<|message|>"
#     end_pattern_pure_output = "<|return|>"
#     pure_output = None
#     if len(gpt_response.split("<|end|>")) != 2:
#         if "<|start|>???<|end|>" in gpt_response:
#             print("Funny Sample -> That weird <|start|>???<|end|> pattern")
#         elif len(gpt_response.split("<|end|>")) == 1:
#             print("Did not finish thinking")
#         else:
#             print(f"This sample funny, splits into list of length {len(gpt_response.split('<|end|>'))}")
#             print(gpt_response)
#     analysis_tokens, output_tokens = gpt_response.split("<|end|>")
#     pattern = re.escape(start_pattern_pure_output) + r"(.*?)" + re.escape(end_pattern_pure_output)
#     match = re.search(pattern, output_tokens)
#     if match:
#         pure_output = match.group(1)
#         output_tokens = output_tokens.replace(pure_output, genz_version)
#         if pure_output in analysis_tokens:
#             analysis_tokens = analysis_tokens.replace(pure_output, genz_version)

#     return {'content': analysis_tokens + "<|end|>" + output_tokens, 'role': 'assistant'}

# # --- MAIN LOOP ---
# custom_training_data = []
# skip_counter = 0
# skip_indexes = []

# for i in tqdm(range(len(df))):
#     row = df.iloc[i]
#     normal, genz = row["normal"], row["gen_z"]
#     ip_for_gpt = generate_gpt_prompt_from_template(normal)
    
#     try:
#         inputs = tokenizer(ip_for_gpt['content'], return_tensors="pt", padding=True, truncation=True).to(model.device)
#         generated = model.generate(**inputs, max_new_tokens=700)
#         op_from_default_gpt = {'content': tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]), 'role': 'assistant'}
        
#         post_process_op = post_process_normal_gpt_response(op_from_default_gpt['content'], genz)
#         row_to_append = {
#             'chosen': [ip_for_gpt, post_process_op],
#             'rejected': [ip_for_gpt, op_from_default_gpt],
#             'score_chosen': round(random.randint(7, 9)+ random.random(), 2),
#             'score_rejected': round(random.randint(1, 2) + random.random(), 2)
#         }
        
#         custom_training_data.append(row_to_append)
#         save_incrementally(row_to_append, output_file) # Immediate Save
        
#     except Exception as e:
#         skip_counter += 1
#         skip_indexes.append(i)
#         continue

# print(f"FIRST PASS SKIP SUMMARY: {skip_counter} skips")

# # --- RETRY LOOP ---
# skips_after_retry, skip_indices_after_retry = 0, []
# print("Re-attempting SKIPS")
# for i in tqdm(skip_indexes):
#     row = df.iloc[i]
#     normal, gen_z = row["normal"], row["gen_z"]
#     ip_for_gpt = generate_gpt_prompt_from_template(normal)
    
#     try:
#         inputs = tokenizer(ip_for_gpt['content'], return_tensors="pt", padding=True, truncation=True).to(model.device)
#         generated = model.generate(**inputs, max_new_tokens=700)
#         op_from_default_gpt = {'content': tokenizer.decode(generated[0][inputs["input_ids"].shape[-1] :]), 'role': 'assistant'}
        
#         post_process_op = post_process_normal_gpt_response(op_from_default_gpt['content'], gen_z)
#         row_to_append = {
#             'chosen': [ip_for_gpt, post_process_op],
#             'rejected': [ip_for_gpt, op_from_default_gpt],
#             'score_chosen': round(random.randint(7, 9)+ random.random(), 2),
#             'score_rejected': round(random.randint(1, 2) + random.random(), 2)
#         }
        
#         custom_training_data.append(row_to_append)
#         save_incrementally(row_to_append, output_file) # Immediate Save
        
#     except Exception as e:
#         skips_after_retry += 1
#         skip_indices_after_retry.append(i)
#         continue

# print(f"FINAL SKIP SUMMARY: {skips_after_retry} remains")