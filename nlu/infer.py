from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
import json
import os


class NLUModel:
    def __init__(self):
        """
        Load pretrained models if available, otherwise fall back to rule-based.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model paths
        self.intent_dir = Path("models/intent").resolve()
        self.slot_dir = Path("models/slot").resolve()

        # --- Check if model folders exist ---
        if self.intent_dir.exists() and self.slot_dir.exists():
            print("✅ Found trained models. Loading...")
            self._load_models()
            self.rule_based = False
        else:
            print("⚠️ Model folders not found. Using rule-based fallback.")
            self.rule_based = True

    # ----------------------------------------------------------
    # Load pretrained models
    # ----------------------------------------------------------
    def _load_models(self):
        # Intent model
        self.intent_tokenizer = AutoTokenizer.from_pretrained(str(self.intent_dir))
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(str(self.intent_dir))
        with open(self.intent_dir / "labels.json", "r", encoding="utf-8") as f:
            self.intent_label2id = json.load(f)["label2id"]
        self.intent_id2label = {v: k for k, v in self.intent_label2id.items()}

        # Slot model
        self.slot_tokenizer = AutoTokenizer.from_pretrained(str(self.slot_dir))
        self.slot_model = AutoModelForTokenClassification.from_pretrained(str(self.slot_dir))
        with open(self.slot_dir / "labels.json", "r", encoding="utf-8") as f:
            self.slot_tag2id = json.load(f)["tag2id"]
        self.slot_id2tag = {v: k for k, v in self.slot_tag2id.items()}

        self.intent_model.to(self.device)
        self.slot_model.to(self.device)
        self.intent_model.eval()
        self.slot_model.eval()

    # ----------------------------------------------------------
    # Main predict method
    # ----------------------------------------------------------
    def predict(self, text: str):
        """Predict intent and slots for a given input text."""

        if self.rule_based:
            return self._rule_based_predict(text)
        else:
            return self._model_predict(text)

    # ----------------------------------------------------------
    # Model-based prediction
    # ----------------------------------------------------------
    @torch.inference_mode()
    def _model_predict(self, text: str):
        # Intent
        intent_inputs = self.intent_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)
        intent_logits = self.intent_model(**intent_inputs).logits
        intent_id = int(torch.argmax(intent_logits, dim=-1).cpu().item())
        intent = self.intent_id2label[intent_id]

        # Slot tagging
        words = text.split()
        slot_inputs = self.slot_tokenizer(
            words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        slot_outputs = self.slot_model(**slot_inputs)
        pred_ids = torch.argmax(slot_outputs.logits, dim=-1)[0].cpu().tolist()
        word_ids = slot_inputs.word_ids(batch_index=0)
        slots = self._decode_slots(words, pred_ids, word_ids)

        return {"intent": intent, "slots": slots}

    # ----------------------------------------------------------
    # Rule-based fallback
    # ----------------------------------------------------------
    def _rule_based_predict(self, text: str):
        text_lower = text.lower()
        if "cancel" in text_lower:
            return {"intent": "CancelBooking", "slots": {}}
        elif "book" in text_lower or "flight" in text_lower:
            return {"intent": "BookFlight", "slots": {}}
        elif "weather" in text_lower:
            return {"intent": "WeatherQuery", "slots": {}}
        elif any(g in text_lower for g in ["hi", "hello", "hey"]):
            return {"intent": "Greeting", "slots": {}}
        elif "bye" in text_lower or "goodbye" in text_lower:
            return {"intent": "Goodbye", "slots": {}}
        else:
            return {"intent": "Unknown", "slots": {}}

    # ----------------------------------------------------------
    # Helper to decode BIO slots
    # ----------------------------------------------------------
    def _decode_slots(self, words, pred_ids, word_ids):
        entities = []
        current_entity = None
        for word_id, tag_id in zip(word_ids, pred_ids):
            if word_id is None or word_id >= len(words):
                continue
            tag = self.slot_id2tag.get(int(tag_id), "O")
            word = words[word_id]
            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"type": tag[2:], "text": word}
            elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
                current_entity["text"] += " " + word
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        if current_entity:
            entities.append(current_entity)
        return {ent["type"]: ent["text"].strip() for ent in entities}
