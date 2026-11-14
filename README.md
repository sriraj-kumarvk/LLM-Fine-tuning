# SLM — Small Language Model Fine-tuning

LoRA fine-tuning for domain-specific adapters on Apple Silicon using MLX.

## Structure

```
SLM/
├── adapters/<domain>/     # LoRA weights + config per domain
├── data/                  # dataset exploration scripts
├── scripts/               # runnable entry points
│   ├── train.py           # example training run
│   ├── evaluate.py        # evaluation / inference
│   └── inference.py       # quick inference test
└── src/
    ├── trainer.py         # Trainer class — main reusable module
    └── modules.py         # Metrics, dataset loading, ProcessedDataset
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file with any required tokens (e.g. `HF_TOKEN=...`).

## Training

```python
from src.trainer import Trainer

Trainer(
    dataset_id="win-wang/Machine_Learning_QA_Collection",
    adapter_path="adapters/ml/",
).train()
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `dataset_id` | required | Hugging Face dataset ID |
| `adapter_path` | required | Output dir for adapter weights |
| `model_path` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Base model |
| `iters` | `200` | Training iterations |
| `batch_size` | `8` | Batch size |
| `learning_rate` | `1e-5` | Adam learning rate |
| `lora_rank` | `8` | LoRA rank |
| `lora_num_layers` | `8` | Number of layers to apply LoRA |
| `max_seq_length` | `1024` | Max token length |
| `plot` | `True` | Show loss curve after training |

`adapter_config.json` is written automatically to `adapter_path` on each run.

## Adding a new domain

```python
Trainer(
    dataset_id="some-user/SQL_QA",
    adapter_path="adapters/sql/",
    iters=300,
    lora_rank=16,
).train()
```

A new `adapters/sql/` folder is created automatically.

## Evaluation / Inference

```bash
python scripts/evaluate.py
```
