import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.trainer import default_loss
from modules import custom_load_hf_dataset, ProcessedDataset
import mlx.core as mx
import math

model_path = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

base_model,tokenizer = load(model_path)
model_ml, tokenizer_ml = load(model_path, 
                            #   adapter_path="adapters/ml/"
                              )

#################################################################
prompt = """
give the natual language query for the following SPARQL query:
SELECT (COUNT(?country) AS ?answer)
WHERE {
    ?country property:member_of resource:Europe .
    ?country property:population ?n .
    FILTER (?n > 10000000)
}"""
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer_ml.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

response = generate(model_ml, tokenizer_ml, prompt=prompt, verbose=True)
#################################################################


# training_args = TrainingArgs(batch_size=8, max_seq_length=1024)

# def evaluate_model(model, tokenizer, label):
#     _, _, test_set_raw = custom_load_hf_dataset(
#         data_id="win-wang/Machine_Learning_QA_Collection",
#         tokenizer=tokenizer,
#         config=training_args,
#         names=("train", "validation", "test"),
#     )
#     test_set = ProcessedDataset(test_set_raw)

#     total_loss = 0.0
#     total_tokens = 0
#     num_batches = 0

#     for batch, lengths in iterate_batches(test_set, batch_size=4,max_seq_length=1024):
#         loss, toks = default_loss(model, batch, lengths)
#         mx.eval(loss, toks)
#         total_loss += loss.item() * toks.item()
#         total_tokens += toks.item()
#         num_batches += 1
#         if num_batches >= 50:  # ~200 examples, enough for eval
#             break

#     avg_loss = total_loss / total_tokens
#     perplexity = math.exp(avg_loss)
#     print(f"{label} — loss: {avg_loss:.4f}, perplexity:{perplexity:.2f}")

# # Base model
# evaluate_model(base_model, tokenizer, "Base model")
# del base_model

# # Fine-tuned model
# evaluate_model(model_ml, tokenizer, "Fine-tuned")