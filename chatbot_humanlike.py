import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from pathlib import Path
from nlu.config import Paths
import json

# ===============================
# NLU WRAPPER
# ===============================
class NLUModel:
    def __init__(self, intent_dir: Path, slot_dir: Path):
        self.intent_tokenizer = AutoTokenizer.from_pretrained(intent_dir)
        self.intent_model = AutoModelForSequenceClassification.from_pretrained(intent_dir)
        with open(intent_dir / "labels.json", "r", encoding="utf-8") as f:
            self.intent_label2id = json.load(f)["label2id"]
        self.intent_id2label = {v: k for k, v in self.intent_label2id.items()}

        self.slot_tokenizer = AutoTokenizer.from_pretrained(slot_dir)
        self.slot_model = AutoModelForTokenClassification.from_pretrained(slot_dir)
        with open(slot_dir / "labels.json", "r", encoding="utf-8") as f:
            self.slot_tag2id = json.load(f)["tag2id"]
        self.slot_id2tag = {v: k for k, v in self.slot_tag2id.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.intent_model.to(self.device)
        self.slot_model.to(self.device)
        self.intent_model.eval()
        self.slot_model.eval()

    @torch.inference_mode()
    def predict(self, text: str):
        # Intent
        intent_inputs = self.intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        intent_logits = self.intent_model(**intent_inputs).logits
        intent_id = int(torch.argmax(intent_logits, dim=-1).cpu().item())
        intent = self.intent_id2label[intent_id]

        # Slot tagging
        words = text.split()
        enc = self.slot_tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True).to(self.device)
        logits = self.slot_model(**enc).logits
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()
        word_ids = enc.word_ids(batch_index=0)

        slots = self.decode_slots(words, preds, word_ids)
        return {"intent": intent, "slots": slots}

    def decode_slots(self, words, pred_ids, word_ids):
        entities = []
        current = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx >= len(words):
                continue
            tag = self.slot_id2tag.get(int(pred_ids[idx]), "O")
            word = words[word_idx]
            if tag.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"type": tag[2:], "text": word}
            elif tag.startswith("I-") and current and current["type"] == tag[2:]:
                current["text"] += " " + word
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)
        return {ent["type"]: ent["text"].strip() for ent in entities}


# ===============================
# CHATBOT WITH CONTEXT MEMORY
# ===============================
class Chatbot:
    def __init__(self, nlu_model: NLUModel):
        self.nlu = nlu_model
        self.context = {"awaiting_pnr": False, "pnr": None}

    def get_response(self, text: str):
        # 1️⃣ If bot is waiting for PNR number
        if self.context["awaiting_pnr"]:
            self.context["pnr"] = text.strip()
            self.context["awaiting_pnr"] = False
            return f"Thanks! Your booking with PNR {self.context['pnr']} has been cancelled successfully ✅"

        # 2️⃣ Predict intent + slots
        pred = self.nlu.predict(text)
        intent = pred["intent"]

        # 3️⃣ Respond based on detected intent
        if intent == "Greeting":
            return "Hi there! How can I help you today?"

        elif intent == "Goodbye":
            return "Goodbye! Have a safe journey!"

        elif intent == "BookFlight":
            return "Sure! Where are you flying from and to?"

        elif intent == "CancelBooking":
            self.context["awaiting_pnr"] = True
            return "Okay, tell me your PNR number."

        elif intent == "WeatherQuery":
            loc = pred["slots"].get("CITY", "your location")
            return f"The weather in {loc} is sunny and pleasant ☀️"

        else:
            return "Sorry, I didn’t quite catch that. Could you repeat?"

# ===============================
# MAIN EXECUTION LOOP
# ===============================
if __name__ == "__main__":
    print("🔹 Loading NLU model...")
    nlu = NLUModel(Paths.intent_model_dir, Paths.slot_model_dir)
    print("🤖 Chatbot is ready! Type 'quit' to exit.\n")

    bot = Chatbot(nlu)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Bot: Goodbye 👋")
            break

        bot_reply = bot.get_response(user_input)
        print("Bot:", bot_reply)
