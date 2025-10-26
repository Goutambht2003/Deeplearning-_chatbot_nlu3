from nlu.infer import NLU
from nlu.config import Paths

def simple_policy(nlu_out):
    intent = nlu_out["intent"]
    slots = nlu_out["slots"]

    if intent == "greeting":
        return "Hello! How can I help you today?"
    if intent == "goodbye":
        return "Bye! Have a great day."
    if intent == "book_flight":
        src = slots.get("FROM_LOC", "somewhere")
        dst = slots.get("TO_LOC", "somewhere")
        date = slots.get("DATE", "a suitable date")
        return f"Okay, booking a flight from {src} to {dst} on {date}. Shall I proceed?"
    if intent == "cancel_flight":
        return "Sure, I can help cancel your flight. Do you have a booking reference?"
    if intent == "weather_query":
        city = slots.get("TO_LOC", "your city")
        date = slots.get("DATE", "today")
        return f"Let me check the weather in {city} for {date}."
    return "Sorry, I didn’t get that. Could you rephrase?"

if __name__ == "__main__":
    nlu = NLU(Paths.intent_model_dir, Paths.slot_model_dir)
    print("Type 'quit' to exit.")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit","exit"}:
            break
        nlu_out = nlu.predict(user)
        print("NLU:", nlu_out)
        print("Bot:", simple_policy(nlu_out))
