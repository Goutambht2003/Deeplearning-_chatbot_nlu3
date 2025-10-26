import gradio as gr
import socket
import json
import random
from nlu.infer import NLUModel

# ---------------------------
# Load NLU model
# ---------------------------
print("🔹 Loading NLU model...")
model = NLUModel()
print("✅ NLU model loaded successfully.")

# ---------------------------
# Load Intents and Slots data
# ---------------------------
def load_jsonl(filepath):
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️ File not found: {filepath}")
    return data

intents_data = load_jsonl("data/intents.jsonl")
slots_data = load_jsonl("data/slots.jsonl")

print(f"✅ Loaded {len(intents_data)} intent samples and {len(slots_data)} slot samples.")

# ---------------------------
# Context memory
# ---------------------------
user_context = {}

# ---------------------------
# Helper functions
# ---------------------------
def generate_pnr():
    """Generate a random PNR string"""
    return "PNR" + str(random.randint(1000, 9999)) + random.choice("ABCDEF")

# ---------------------------
# Slot extraction with city list
# ---------------------------
city_list = set()
for entry in slots_data:
    for i, label in enumerate(entry.get("labels", [])):
        if label in ["B-FROM_LOC", "B-TO_LOC", "B-CITY"]:
            token = entry["tokens"][i].capitalize()
            city_list.add(token)

def extract_slots(message):
    """Extract FROM, TO, and CITY info based on detected city names"""
    message_lower = message.lower()
    detected_cities = [city for city in city_list if city.lower() in message_lower]
    from_city = detected_cities[0] if len(detected_cities) >= 1 else None
    to_city = detected_cities[1] if len(detected_cities) >= 2 else None
    city_name = detected_cities[0] if detected_cities else None
    return from_city, to_city, city_name

# ---------------------------
# Chatbot response
# ---------------------------
def chatbot_response(message, history):
    user_id = "default_user"
    intent_data = model.predict(message)
    intent = intent_data.get("intent", "Unknown")

    from_city, to_city, city_name = extract_slots(message)
    context = user_context.get(user_id, {})
    reply = ""

    print(f"🧠 Intent={intent}, from={from_city}, to={to_city}, city={city_name}")

    if intent == "Greeting":
        reply = "👋 Hello there! How can I assist you today?"

    elif intent == "BookFlight":
        if from_city and to_city:
            pnr = generate_pnr()
            reply = f"✅ Your ticket from {from_city} to {to_city} is booked. Your PNR number is {pnr}."
            user_context[user_id] = {"pnr": pnr, "from": from_city, "to": to_city}
        else:
            user_context[user_id] = {"last_intent": "BookFlight", "from": from_city, "to": to_city}
            reply = "Sure! Could you tell me both the source and destination cities?"

    elif intent == "CancelBooking":
        if context.get("pnr"):
            pnr = context["pnr"]
            reply = f"🛑 Your booking with PNR {pnr} has been canceled successfully."
            user_context[user_id] = {}
        else:
            user_context[user_id] = {"awaiting_pnr": True}
            reply = "Please tell me your PNR number to cancel the booking."

    elif context.get("awaiting_pnr"):
        pnr_input = message.strip().upper()
        user_context[user_id] = {}
        reply = f"🛑 Your booking with PNR {pnr_input} has been canceled successfully."

    elif intent == "WeatherQuery":
        if city_name:
            reply = f"🌤️ The weather in {city_name} is pleasant today."
        else:
            reply = "Please tell me which city's weather you'd like to know."

    elif intent == "Goodbye":
        reply = "👋 Goodbye! Have a great day!"

    else:
        reply = "🤔 I'm not sure I understand. Could you please rephrase?"

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    return new_history

# ---------------------------
# Get local IP for LAN access
# ---------------------------
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ---------------------------
# Launch Gradio Interface
# ---------------------------
with gr.Blocks(title="AI Flight Assistant") as demo:
    gr.Markdown("# ✈️ AI Flight Assistant (Dynamic Data-driven)")
    gr.ChatInterface(
        fn=chatbot_response,
        type="messages",
        examples=[
            "Book a flight from Delhi to Mumbai tomorrow",
            "Cancel my booking",
            "Is it raining in Delhi?",
            "Hi there",
        ],
    )

local_ip = get_local_ip()
print("\n🌍 Chatbot starting...")
print(f"🔗 Access locally  : http://127.0.0.1:7860")
print(f"🔗 Access on LAN   : http://{local_ip}:7860\n")

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
