import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv()

from src.trainer import Trainer

Trainer(
    dataset_id="win-wang/Machine_Learning_QA_Collection",
    adapter_path="adapters/ml/",
).train()
