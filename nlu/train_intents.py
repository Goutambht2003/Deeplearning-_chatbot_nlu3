from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from sklearn.metrics import classification_report, accuracy_score
from nlu.config import Paths, IntentConfig
from nlu.utils import prepare_intent_splits, build_label_map, inverse_label_map, save_json

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, label2id):
        self.texts = list(texts)
        self.labels = [label2id[l] for l in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        t = self.texts[idx]
        enc = self.tokenizer(
            t, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train():
    paths = Paths()
    cfg = IntentConfig()
    paths.intent_model_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = prepare_intent_splits(paths.data / "intents.jsonl")
    label2id = build_label_map(train_df["label"].tolist())
    id2label = inverse_label_map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )

    train_ds = IntentDataset(train_df["text"], train_df["label"], tokenizer, cfg.max_len, label2id)
    test_ds  = IntentDataset(test_df["text"],  test_df["label"],  tokenizer, cfg.max_len, label2id)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

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
        print(f"[Intent] Epoch {epoch+1}/{cfg.epochs} - loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].numpy().tolist()
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend(pred); trues.extend(labels)

    acc = accuracy_score(trues, preds)
    print("Intent Accuracy:", acc)
    print(classification_report(trues, preds, target_names=[id2label[i] for i in sorted(id2label)]))

    # Save
    model.save_pretrained(paths.intent_model_dir)
    tokenizer.save_pretrained(paths.intent_model_dir)
    save_json({"label2id": label2id}, paths.intent_model_dir / "labels.json")
    print(f"Saved intent model to: {paths.intent_model_dir}")

if __name__ == "__main__":
    train()
