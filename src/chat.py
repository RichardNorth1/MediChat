import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import colorama 
colorama.init()
from colorama import Fore, Style
import language_tool_python
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random
from . import data_loader


# Load the fine-tuned model and tokenizer
model_path = "data/models/fine_tuned_model"
tranformer_tokenizer = AutoTokenizer.from_pretrained(model_path)
transformer_model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the trained deep learning model
deep_learning_model = keras.models.load_model('data/models/chat_model.h5')

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

def correct_grammar(input_text):
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    return corrected_text

def get_response(intents, intent):
    for data in intents:
        for i in data['intents']:
            if i['tag'] == intent:
                return np.random.choice(i['responses'])
    return "Sorry, I didn't understand that."


def predict_intent(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)
    predictions = deep_learning_model.predict(padded_sequences)
    predicted_label = np.argmax(predictions)
    intent = label_encoder.inverse_transform([predicted_label])[0]
    return intent

def classify_escalation(input_text):
    inputs = tranformer_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Modify this line to set a higher threshold (e.g., 0.7)
    if confidence >= 0.57 and predicted_class == 1:
        return predicted_class, confidence
    else:
        return 0, confidence  # Default to non-critical if confidence is low



def handle_chat(input_text):
    # Correct grammar in the input text
    corrected_inp = correct_grammar(input_text)

    # Check if escalation is needed
    predicted_class, confidence = classify_escalation(input_text)
    if predicted_class == 1:
        return {"message": f"Escalating the conversation to a human agent due to detected critical input (Confidence: {confidence:.2f}).", "escalate": True}

    # Predict the intent of the user input
    intent = predict_intent(corrected_inp)

    # Get response based on the intent
    response_message = get_response(all_intents, intent)
    return {"message": response_message, "escalate": False}


def chat():
    while True:
        user_input = input(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL)
        response = handle_chat(user_input)
        print(Fore.YELLOW + "ChatBot: " + response['message'] + Style.RESET_ALL)  # Fixed this line to use the 'message' key


if __name__ == "__main__":
    chat()
