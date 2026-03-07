# This file is fully AI generated. I gave the format of the logging file and asked Gemini to write the code for plotting all the metrics.

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
# Replace 'trainer_state.json' with your actual file path
file_path = 'helper_utils/outputs/gpt-oss-stackover-aligned-DPO-1e-ckpt/trainer_state.json' 
with open(file_path, 'r') as f:
    data = json.load(f)

# 2. Extract log history into a DataFrame
df = pd.DataFrame(data['log_history'])

# Filter out evaluation logs if they exist (they lack the 'step' key usually)
df = df.dropna(subset=['loss'])

# 3. Setup Plotting Style
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f"DPO Training Metrics - {data.get('model_name', 'Gemma-3-1B-it')}", fontsize=16)

# --- Plot 1: Loss ---
sns.lineplot(ax=axes[0, 0], data=df, x='step', y='loss', color='royalblue')
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_ylabel("Cross Entropy Loss")

# --- Plot 2: Rewards (Chosen vs Rejected) ---
sns.lineplot(ax=axes[0, 1], data=df, x='step', y='rewards/chosen', label='Chosen', color='green')
sns.lineplot(ax=axes[0, 1], data=df, x='step', y='rewards/rejected', label='Rejected', color='red')
axes[0, 1].set_title("Rewards Comparison")
axes[0, 1].set_ylabel("Reward Value")
axes[0, 1].legend()

# --- Plot 3: Reward Margin ---
# The margin is the difference between chosen and rejected. 
# It should ideally increase over time.
sns.lineplot(ax=axes[1, 0], data=df, x='step', y='rewards/margins', color='purple')
axes[1, 0].set_title("Reward Margin (Chosen - Rejected)")
axes[1, 0].set_ylabel("Margin")

# --- Plot 4: Accuracy ---
sns.lineplot(ax=axes[1, 1], data=df, x='step', y='rewards/accuracies', color='orange')
axes[1, 1].set_title("Reward Accuracy")
axes[1, 1].set_ylabel("Accuracy (%)")
axes[1, 1].set_ylim(0, 1.1) # Accuracy is 0 to 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('GPT-OSS-20B-it-Training-Logs-stackoverflow.png', dpi=300, bbox_inches='tight')