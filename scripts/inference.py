import json
import os
from typing import Dict, List, Optional, Tuple, Union

import os
from dotenv import load_dotenv
load_dotenv()

import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load


model_path = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
model, tokenizer = load(model_path)

prompt = """
Here's a sql database schema
department: Department_ID (number), Name (text), Creation (text), Ranking (number), 
Budget_in_Billions (number), Num_Employees (number) | head: head_ID (number), name (text), 
born_state (text), age (number) | management: department_ID (number), head_ID (number), temporary_acting (text)

now write a sql query to find the names of all departments that have more than 100 employees and a 
budget greater than 1 billion dollars, along with the name of the head of each department. 
The results should be ordered by the department's ranking in ascending order.
"""
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(response)


"""

"""

# ##################################################
# import os
# adapter_path = "adapters"
# os.makedirs(adapter_path, exist_ok=True)
# adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
# adapter_file_path = os.path.join(adapter_path, "adapters.safetensors")

# ##################################################

# import json
# lora_config = {
#     "num_layers": 8,
#     "lora_parameters": {
#         "rank": 8,
#         "scale": 20.0,
#         "dropout": 0.0,
#     },
# }

# with open(adapter_config_path, "w") as f:
#     json.dump(lora_config, f, indent=4)
