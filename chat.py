
import random
import json
import re
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

state = {
    "expecting_matriculation_number": False,
    "matriculation_number": None
}

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
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)


#Before Changes to code structure --> 03:15 15th July, 2024
# import random
# import json

# import torch

# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Sam"

# def get_response(msg):
#     sentence = tokenize(msg)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
    
#     return "I do not understand..."


# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         # sentence = "do you use credit cards?"
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         print(resp)

