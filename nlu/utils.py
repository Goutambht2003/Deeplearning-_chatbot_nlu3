import json, random
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def prepare_intent_splits(data_path: Path, test_size=0.25, seed=42):
    rows = read_jsonl(data_path)
    df = pd.DataFrame(rows)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def build_label_map(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}

def inverse_label_map(label2id: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in label2id.items()}

def prepare_slot_splits(data_path: Path, test_size=0.2, seed=42):
    rows = read_jsonl(data_path)
    random.seed(seed)
    random.shuffle(rows)
    n_test = max(1, int(len(rows) * test_size))
    test = rows[:n_test]
    train = rows[n_test:]
    return train, test

def align_labels_with_subtokens(labels, word_ids):
    # Map word-level BIO labels to sub-tokens (DistilBERT uses WordPiece)
    aligned = []
    prev_word = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)  # ignore
        else:
            label = labels[word_id]
            if word_id != prev_word:
                aligned.append(label)
            else:
                # For subwords inside a word, convert B-XXX to I-XXX
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                aligned.append(label)
            prev_word = word_id
    return aligned

def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)
