from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    data: Path = root / "data"
    artifacts: Path = root / "artifacts"          # models + label maps
    intent_model_dir: Path = artifacts / "intent"
    slot_model_dir: Path = artifacts / "slots"

@dataclass
class IntentConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 64
    batch_size: int = 16
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

@dataclass
class SlotConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 64
    batch_size: int = 16
    epochs: int = 6
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
