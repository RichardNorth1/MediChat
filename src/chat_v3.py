import os
import json
from difflib import get_close_matches

# Load QA Pairs
qa_file = "./data/nausea_qa_pairs.json"  # Update with the correct file path

def load_qa_pairs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["qa_pairs"]

qa_pairs = load_qa_pairs(qa_file)
questions = [pair["question"] for pair in qa_pairs]

def find_best_match(user_question):
    """
    Find the closest stored question and return its answer.
    """
    best_matches = get_close_matches(user_question, questions, n=1, cutoff=0.7)  # Adjust cutoff as needed
    if best_matches:
        matched_question = best_matches[0]
        for pair in qa_pairs:
            if pair["question"] == matched_question:
                return pair["answer"]
    
    return "I don't have an answer for that. Can you try rephrasing?"

def chatbot():
    print("🤖 MediChat Bot: Ask me anything about nausea! Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("🤖 MediChat Bot: Goodbye!")
            break
        
        response = find_best_match(user_input)
        print(f"🤖 MediChat Bot: {response}")

if __name__ == "__main__":
    chatbot()
