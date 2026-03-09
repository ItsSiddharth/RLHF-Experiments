import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from unsloth import FastLanguageModel
import time

max_seq_length = 2048 
model_name = "helper_utils/outputs/gemma3-1b-stackover-aligned-DPO-1e-ckpt"

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
    start_time = time.time()
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
        print(f"Time taken for DPO Aligned Gemma3: {time.time() - start_time}")
        ft_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    start_time = time.time()
    # B. Vanilla Output
    with model.disable_adapter():
        with torch.no_grad():
            outputs_v = model.generate(**inputs, max_new_tokens = 2048, use_cache = True, temperature = 0.1, repetition_penalty = 1.15)
            print(f"Time taken for Vanilla Gemma3: {time.time() - start_time}")
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

"""
    [QUERY]: How to filter a dataframe by date range?
--- VANILLA GEMMA 3 --- 
```python
import pandas as pd

# Assuming your DataFrame is named 'df'
df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
                   'Value':[10, 20, 30, 40, 50]})


# Filter for dates from 2023-01-01 to 2023-12-31
filtered_df = df[pd.to_datetime(df['Date'])]

# Print the filtered DataFrame
print(filtered_df)
```

**Explanation:**

* **`import pandas as pd`**: Imports the Pandas library, which provides data manipulation tools.
* **`df = pd.DataFrame(...)`**: Creates a sample DataFrame (replace this with your actual DataFrame).  It includes a 'Date' column.
* **`df['Date'] = pd.to_datetime(df['Date'])`**: Converts the 'Date' column into datetime objects. This is crucial because Pandas needs to understand the date format correctly when filtering.
* **`df[pd.to_datetime(df['Date'])]`**: This is the core of the filtering operation. It uses boolean indexing to select rows where the `Date` column falls within the specified range (`2023-01-01` to `2023-12-31`).  Pandas automatically handles the comparison of datetime objects.
* **`print(filtered_df)`**: Prints the resulting DataFrame containing only the desired rows.
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
--- DPO ALIGNED GEMMA 3 --- 
```python
import pandas as pd

# Assuming your DataFrame is named 'df'
df = pd.DataFrame({'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
                'Value':[10, 20, 30, 40, 50]})


filtered_df = df[pd.to_datetime(df['Date'])]  # Convert 'Date' column to datetime objects first

print(filtered_df)
```

**Explanation:**

* **`df['Date'] = pd.to_datetime(df['Date'])`**: This line converts the 'Date' column into datetime objects. Pandas automatically handles this if the string format is consistent. If your dates aren't consistently formatted, you might need more complex parsing logic using `pd.to_datetime()` or other methods for proper conversion.
* **`df[pd.to_datetime(df['Date'])]`**: This uses boolean indexing (also known as "loc" or "bracket notation") to select rows where the 'Date' column matches the specified date range.  `pd.to_datetime(df['Date'])` converts the 'Date' column to datetime objects so that we can compare them. The square brackets `[]` create a copy of the DataFrame containing only the desired rows.

This approach efficiently filters the DataFrame based on the date range provided.  The result will be a new DataFrame containing only the rows matching the date criteria.
============================================================
"""