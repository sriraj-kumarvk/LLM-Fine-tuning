import sys, os; sys.path.insert(0, os.path.dirname(__file__))

import json
import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner import TrainingArgs, linear_to_lora_layers, train

from modules import Metrics, custom_load_hf_dataset, ProcessedDataset


class Trainer:
    def __init__(
        self,
        dataset_id: str,
        adapter_path: str,
        model_path: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        # LoRA
        lora_num_layers: int = 8,
        lora_rank: int = 8,
        lora_scale: float = 2.0,
        lora_dropout: float = 0.0,
        # training
        batch_size: int = 8,
        iters: int = 200,
        val_batches: int = 25,
        steps_per_report: int = 10,
        steps_per_eval: int = 50,
        steps_per_save: int = 100,
        max_seq_length: int = 1024,
        learning_rate: float = 1e-5,
        grad_checkpoint: bool = False,
        grad_accumulation_steps: int = 1,
        dataset_split_names: tuple = ("train", "validation", "test"),
        plot: bool = True,
    ):
        self.dataset_id = dataset_id
        self.adapter_path = adapter_path
        self.model_path = model_path
        self.lora_num_layers = lora_num_layers
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale
        self.lora_dropout = lora_dropout
        self.batch_size = batch_size
        self.iters = iters
        self.val_batches = val_batches
        self.steps_per_report = steps_per_report
        self.steps_per_eval = steps_per_eval
        self.steps_per_save = steps_per_save
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.grad_checkpoint = grad_checkpoint
        self.grad_accumulation_steps = grad_accumulation_steps
        self.dataset_split_names = dataset_split_names
        self.plot = plot

    def train(self) -> Metrics:
        os.makedirs(self.adapter_path, exist_ok=True)

        model, tokenizer = load(self.model_path)

        lora_config = {
            "num_layers": self.lora_num_layers,
            "lora_parameters": {
                "rank": self.lora_rank,
                "scale": self.lora_scale,
                "dropout": self.lora_dropout,
            },
        }
        adapter_config_path = os.path.join(self.adapter_path, "adapter_config.json")
        with open(adapter_config_path, "w") as f:
            json.dump(lora_config, f, indent=4)

        model.freeze()
        linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"])
        num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
        print(f"Number of trainable parameters: {num_train_params}")
        model.train()

        training_args = TrainingArgs(
            batch_size=self.batch_size,
            iters=self.iters,
            val_batches=self.val_batches,
            steps_per_report=self.steps_per_report,
            steps_per_eval=self.steps_per_eval,
            steps_per_save=self.steps_per_save,
            max_seq_length=self.max_seq_length,
            adapter_file=os.path.join(self.adapter_path, "adapters.safetensors"),
            grad_checkpoint=self.grad_checkpoint,
            grad_accumulation_steps=self.grad_accumulation_steps,
        )

        metrics = Metrics()

        train_set_raw, val_set_raw, _ = custom_load_hf_dataset(
            data_id=self.dataset_id,
            tokenizer=tokenizer,
            config=training_args,
            names=self.dataset_split_names,
        )

        train_set = ProcessedDataset(train_set_raw)
        val_set = ProcessedDataset(val_set_raw)
        print(f"Train set size: {len(train_set)}")

        train(
            model=model,
            args=training_args,
            optimizer=optim.Adam(learning_rate=self.learning_rate),
            train_dataset=train_set,
            val_dataset=val_set,
            training_callback=metrics,
        )

        if self.plot:
            train_its, train_losses = zip(*metrics.train_losses)
            val_its, val_losses = zip(*metrics.val_losses)
            plt.plot(train_its, train_losses, "-o", label="Train")
            plt.plot(val_its, val_losses, "-o", label="Validation")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        return metrics
