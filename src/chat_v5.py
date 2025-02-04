import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import colorama
colorama.init()
from colorama import Fore, Style
import language_tool_python

# Set up paths
escalation_model_path = "data/models/fine_tuned_model"

# Load QA Pairs
qa_file = "./data/qa_pairs"

# load models and tokenizers
escalation_tokenizer = AutoTokenizer.from_pretrained(escalation_model_path)
escalation_model = AutoModelForSequenceClassification.from_pretrained(escalation_model_path)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_qa_pairs(json_path):
    """Load QA pairs from the given JSON file."""
    all_qa_pairs = []
    for filename in os.listdir(json_path):
        if filename.endswith('.json'):
            with open(os.path.join(json_path, filename)) as file:
                data = json.load(file)
                all_qa_pairs.extend(data["qa_pairs"])
    return all_qa_pairs

qa_pairs = load_qa_pairs(qa_file)

# Extract questions and answers
questions = [pair["question"] for pair in qa_pairs]
answers = [pair["answer"] for pair in qa_pairs]

# Convert stored questions to embeddings
stored_embeddings = model.encode(questions, convert_to_tensor=True)

def get_answer(query):
    """Get the best answer for the query."""    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)
    
    # Find the index of the highest score
    best_match_idx = torch.argmax(cosine_scores)
    
    # Return the corresponding answer
    return answers[best_match_idx]

def correct_grammar(input_text):
    """Correct grammatical errors in the input text."""
    tool = language_tool_python.LanguageTool('en-GB')
    matches = tool.check(input_text)
    corrected_text = language_tool_python.utils.correct(input_text, matches)
    return corrected_text

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
        
        if confidence >= 0.55 and predicted_class == 1:
            return predicted_class, confidence
        else:
            return 0, confidence
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

    # Get response based on the intent
    response_message = get_answer(corrected_inp)
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