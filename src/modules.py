from typing import List, Tuple, Dict, Union
from transformers import PreTrainedTokenizer
from mlx_lm.tuner import  datasets

class Metrics:
    def __init__(self) -> None:
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.val_losses.append((info["iteration"], info["val_loss"]))


def custom_load_hf_dataset(
    data_id: str,
    tokenizer: PreTrainedTokenizer,
    config,
    names: Tuple[str, str, str] = ("train", "valid", "test"),
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        train, valid, test = [
            (
                datasets.create_dataset(dataset[n], tokenizer,config)
                if n in dataset.keys()
                else []
            )
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


class ProcessedDataset:
      def __init__(self, dataset):
          self._dataset = dataset

      def __len__(self):
          return len(self._dataset)

      def __getitem__(self, idx):
          return self._dataset.process(self._dataset._data[idx])
