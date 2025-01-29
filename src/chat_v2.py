import numpy as np
import colorama
colorama.init()
from colorama import Fore, Style
import language_tool_python
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from . import data_loader

# Set up paths
intent_model_path = "data/models/intent_model"
escalation_model_path = "data/models/fine_tuned_model"
intents_folder = 'data/intents'

# Load models and tokenizers
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path)

escalation_tokenizer = AutoTokenizer.from_pretrained(escalation_model_path)
escalation_model = AutoModelForSequenceClassification.from_pretrained(escalation_model_path)

# Load label mapping for intent classification
with open(f"{intent_model_path}/label_mapping.json", "r") as f:
    label_mapping = json.load(f)
id2label = label_mapping["id2label"]

# Load all intents from the intents folder
all_intents = data_loader.load_intents(intents_folder)

def correct_grammar(input_text):
    """Correct grammatical errors in the input text."""
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    return corrected_text

def get_response(intents, intent):
    """Get a response from the intents list based on the predicted intent."""
    for data in intents:
        for i in data['intents']:
            if i['tag'] == intent:
                return np.random.choice(i['responses'])
    return "Sorry, I didn't understand that"

def predict_intent(input_text):
    """Predict the intent of the given input text."""
    try:
        inputs = intent_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = intent_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = id2label[str(predicted_class)]
        return predicted_label
    except Exception as e:
        print(f"Error in predict_intent: {e}")
        return "unknown"

def classify_escalation(input_text):
    """Classify whether the input text requires escalation."""
    try:
        inputs = escalation_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = escalation_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Escalate if confidence is above 0.57 and critical class (1) is detected
        if confidence >= 0.5 and predicted_class == 1:
            return predicted_class, confidence
        else:
            return 0, confidence  # Default to non-critical if confidence is low
    except Exception as e:
        print(f"Error in classify_escalation: {e}")
        return 0, 0.0

def handle_chat(input_text):
    """Handle the chat by predicting the intent and providing a response."""
    # Correct grammar in the input text
    corrected_inp = correct_grammar(input_text)

    # Check if escalation is needed
    predicted_class, confidence = classify_escalation(corrected_inp)
    if predicted_class == 1:
        return {"message": f"Escalating the conversation to a human agent due to detected critical input (Confidence: {confidence:.2f}).", "escalate": True}

    # Predict the intent of the user input
    intent = predict_intent(corrected_inp)

    # Get response based on the intent
    response_message = get_response(all_intents, intent)
    return {"message": response_message, "escalate": False}

def chat():
    """Start the chatbot interaction loop."""
    print("Chatbot is running! Type 'exit' to quit.")
    while True:
        user_input = input(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL)
        if user_input.lower() in ['exit', 'quit']:
            print(Fore.YELLOW + "ChatBot: Goodbye!" + Style.RESET_ALL)
            break
        response = handle_chat(user_input)
        print(Fore.YELLOW + "ChatBot: " + response['message'] + Style.RESET_ALL)

if __name__ == "__main__":
    chat()
