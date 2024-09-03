import os
import random
import json
import re
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Determine the device to run the model on (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the intents.json file
intents_path = os.path.join(script_dir, '../public/intents.json')

# Load the intents file
with open(intents_path, 'r') as json_data:
    intents = json.load(json_data)

# Load the model's state
FILE = os.path.join(script_dir, 'data.pth')
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize and load the neural network model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Define bot name and initial state
bot_name = "Chris"

state = {
    "recording_consent": None,
    "expecting_matriculation_number": False,
    "matriculation_number": None,
    "complaint": "",
    "user_satisfied": False
}

def start_chat():
    print(f"{bot_name}: Hello! I’m here to help you with your complaints.")
    print(f"{bot_name}: Do you want this chat to be recorded for quality and safety purposes? (yes/no)")
    while True:
        response = input("You: ").strip().lower()
        if response == 'yes':
            state["recording_consent"] = True
            print(f"{bot_name}: Thank you! Your chat will be recorded.")
            break
        elif response == 'no':
            state["recording_consent"] = False
            print(f"{bot_name}: No problem! This chat will not be recorded.")
            break
        else:
            print(f"{bot_name}: I didn't understand that. Please type 'yes' or 'no'.")

def get_complaint():
    print(f"{bot_name}: Please describe your complaint.")
    complaint = input("You: ").strip()
    state["complaint"] = complaint

    if state["recording_consent"]:
        # Save the complaint to the database or a file
        save_complaint(complaint)

    handle_complaint(complaint)

def handle_complaint(complaint):
    # Simulate resolving the complaint
    print(f"{bot_name}: Thank you for your complaint. We’re processing it now.")
    # Here you can integrate actual resolution logic or fetch responses from intents
    print(f"{bot_name}: Your complaint has been resolved with the following action: [Simulated Response]")

    check_satisfaction()

def check_satisfaction():
    print(f"{bot_name}: Are you satisfied with the resolution? (yes/no)")
    while True:
        response = input("You: ").strip().lower()
        if response == 'yes':
            state["user_satisfied"] = True
            print(f"{bot_name}: We’re glad to hear that! If you need anything else, just let us know.")
            break
        elif response == 'no':
            state["user_satisfied"] = False
            escalate_to_admin(state["complaint"])
            break
        else:
            print(f"{bot_name}: I didn't understand that. Please type 'yes' or 'no'.")

def escalate_to_admin(complaint):
    print(f"{bot_name}: We’re sorry you’re not satisfied. Your complaint will be escalated to the admin panel for further review.")
    # Implement actual logic to send the complaint to the admin panel
    send_to_admin_panel(complaint)

def save_complaint(complaint):
    # Logic to save the complaint to a database or a file
    # For demonstration, we'll append it to a file named 'complaints_log.txt'
    log_path = os.path.join(script_dir, 'complaints_log.txt')
    with open(log_path, 'a') as log_file:
        log_file.write(f"Complaint: {complaint}\n")
    print(f"{bot_name}: Your complaint has been saved.")

def send_to_admin_panel(complaint):
    # Logic to send the complaint to the admin panel
    # This could be an API call, email, or any other method
    # For demonstration, we'll print a message
    print(f"{bot_name}: Sending complaint to the admin panel...")
    # Example: You can use requests to send an API call
    # import requests
    # api_url = "https://youradminpanel.com/api/complaints"
    # payload = {"complaint": complaint}
    # response = requests.post(api_url, json=payload)
    # if response.status_code == 200:
    #     print(f"{bot_name}: Complaint successfully sent to the admin panel.")
    # else:
    #     print(f"{bot_name}: Failed to send complaint to the admin panel.")

def get_response(msg):
    global state
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    
    if state["expecting_matriculation_number"]:
        state["expecting_matriculation_number"] = False
        state["matriculation_number"] = msg
        return f"Hello {msg}, what issue are you facing?"

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "matriculated_complaint":
                    state["expecting_matriculation_number"] = True
                return random.choice(intent['responses'])
    
    return "I do not understand..."

if __name__ == "__main__":
    start_chat()
    get_complaint()
