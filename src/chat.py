import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import colorama 
colorama.init()
from colorama import Fore, Style
import random
import language_tool_python
import data_loader

# Load the trained deep learning model
model = keras.models.load_model('data/models/chat_model.h5')

# Load the tokenizer object
with open('data/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder object
with open('data/models/label_encoder.pickle', 'rb') as enc:
    label_encoder = pickle.load(enc)

# Path to the intents folder
intents_folder = 'data/intents'

# List to hold all intents
all_intents = data_loader.load_intents(intents_folder)

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def symptom_checker(input_text):
    escalation_symptoms = [
        "chest pain", "shortness of breath", "severe headache", "high fever",
        "uncontrolled bleeding", "loss of consciousness", "severe allergic reaction",
        "persistent vomiting", "severe abdominal pain", "confusion"
    ]
    detected_symptoms = [symptom for symptom in escalation_symptoms if symptom in input_text.lower()]
    return detected_symptoms

def correct_grammar(input_text):
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    return corrected_text

def escalation_check(input_text):
    escalate_text = [
            "I need to speak to a human",
            "Can I talk to a real person?",
            "I want to escalate this issue",
            "I need help from a human",
            "Can you transfer me to a human agent?",
            "I am not satisfied with this response",
            "I need more assistance",
            "This is not helping",
            "I need to talk to someone",
            "Can you connect me to a human?",
            "I need to speak with a representative",
            "I want to talk to a human",
            "I need human assistance",
            "Can I get help from a person?",
            "I need to escalate this",
            "I need to speak to customer service",
            "I want to talk to a support agent",
            "Can you escalate this issue?",
            "I need to speak to someone in charge",
            "I need to talk to a manager",
            "can i speak to a doctor",
            "can i speak to a nurse",
            "can i speak to a physician",
            "i need to speak to someone"
        ]
    return any(escalate_phrase in input_text.lower() for escalate_phrase in escalate_text)

def get_response(intents, intent):
    for data in intents:
        for i in data['intents']:
            if i['tag'] == intent:
                return np.random.choice(i['responses'])
    return "Sorry, I didn't understand that."


def predict_intent(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)
    predictions = model.predict(padded_sequences)
    predicted_label = np.argmax(predictions)
    intent = label_encoder.inverse_transform([predicted_label])[0]
    return intent


def handle_chat(input_text):
    # Correct grammar in the input text
    corrected_inp = correct_grammar(input_text)

    # Check for symptoms that require escalation
    detected_symptoms = symptom_checker(corrected_inp)
    if detected_symptoms:
        return f"Escalating the conversation to a human agent due to detected symptoms: {', '.join(detected_symptoms)}."

    # Predict the intent of the user input
    intent = predict_intent(corrected_inp)

    if escalation_check(corrected_inp):
        return "Escalating the conversation to a human agent."
    else:
        return get_response(all_intents, intent)

def chat():
    while True:
        user_input = input(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL)
        response = handle_chat(user_input)
        print(Fore.YELLOW + "ChatBot: " + response + Style.RESET_ALL)
if __name__ == "__main__":
    chat()