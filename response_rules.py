# response_rules.py

RESPONSE_RULES = {
    "CancelBooking": {
        "slots_required": ["PNR"],  # list slots required for this intent
        "prompt_missing": "Okay, Tell me your PNR no",
        "response_template": "Your ticket with PNR {PNR} has been canceled."
    },
    "BookFlight": {
        "slots_required": ["FROM_LOC", "TO_LOC", "DATE"],
        "prompt_missing": "Where and when do you want to fly?",
        "response_template": "Booking a flight from {FROM_LOC} to {TO_LOC} on {DATE}."
    },
    "WeatherQuery": {
        "slots_required": ["LOC", "DATE"],
        "prompt_missing": "Which city and date do you want the weather for?",
        "response_template": "Weather in {LOC} on {DATE} is sunny."
    }
}
