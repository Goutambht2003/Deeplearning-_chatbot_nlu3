from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report
from nlu.config import Paths, SlotConfig
from nlu.utils import read_jsonl, prepare_slot_splits, save_json


# --------------------------------------------------
# Safe Alignment Function
# --------------------------------------------------
def align_labels_with_subtokens(labels, word_ids):
    """
    Align original word-level labels with subword tokens.

    Each subword that belongs to a given word gets the same label.
    Special tokens (None in word_ids) are assigned -100.
    """
    aligned = []
    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned.append(-100)  # Ignore [CLS], [SEP], [PAD]
        elif word_idx != prev_word_idx:
            # new word
            if word_idx < len(labels):
                aligned.append(labels[word_idx])
            else:
                # happens if tokenizer truncates
                aligned.append(-100)
        else:
            # same word → same label for subwords
            if word_idx < len(labels):
                aligned.append(labels[word_idx])
            else:
                aligned.append(-100)
        prev_word_idx = word_idx
    return aligned


# --------------------------------------------------
# Dataset Class
# --------------------------------------------------
class SlotDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len, tag2id):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        tokens = self.rows[idx]["tokens"]
        labels = self.rows[idx]["labels"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = enc.word_ids(batch_index=0)

        # align string labels first
        aligned_tags = align_labels_with_subtokens(labels, word_ids)

        # convert string tags to ids; keep -100 as ignore index
        aligned_ids = [
            self.tag2id[t] if t in self.tag2id else -100
            if isinstance(t, str) else t for t in aligned_tags
        ]

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(aligned_ids, dtype=torch.long)

        return item


# --------------------------------------------------
# Classification Report Helper
# --------------------------------------------------
def compute_cr(true_tags, pred_tags, id2tag):
    true_flat, pred_flat = [], []
    for t_seq, p_seq in zip(true_tags, pred_tags):
        for t, p in zip(t_seq, p_seq):
            if t == -100:
                continue
            true_flat.append(id2tag[t])
            pred_flat.append(id2tag[p])
    report = classification_report(true_flat, pred_flat, zero_division=0)
    return report


# --------------------------------------------------
# Training Function
# --------------------------------------------------
def train():
    paths = Paths()
    cfg = SlotConfig()
    paths.slot_model_dir.mkdir(parents=True, exist_ok=True)

    # Load and split data
    data_path = paths.data / "slots.jsonl"
    train_rows, test_rows = prepare_slot_splits(data_path)

    # Build tag map from train set
    tags = sorted({t for r in train_rows for t in r["labels"]})
    tag2id = {t: i for i, t in enumerate(tags)}
    id2tag = {v: k for k, v in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(tag2id),
        id2label=id2tag,
        label2id=tag2id
    )

    train_ds = SlotDataset(train_rows, tokenizer, cfg.max_len, tag2id)
    test_ds = SlotDataset(test_rows, tokenizer, cfg.max_len, tag2id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # -------------------------------
    # Training Loop
    # -------------------------------
    model.train()
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"[Slots] Epoch {epoch + 1}/{cfg.epochs} - loss: {total_loss / len(train_loader):.4f}")

    # -------------------------------
    # Evaluation
    # -------------------------------
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].cpu().numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            all_true.extend(labels)
            all_pred.extend(pred)

    print(compute_cr(all_true, all_pred, id2tag))

    # Save model & tokenizer
    model.save_pretrained(paths.slot_model_dir)
    tokenizer.save_pretrained(paths.slot_model_dir)
    save_json({"tag2id": tag2id}, paths.slot_model_dir / "labels.json")
    print(f"✅ Saved slot model to: {paths.slot_model_dir}")


if __name__ == "__main__":
    train()
