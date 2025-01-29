import os
import json
import torch
from sentence_transformers import SentenceTransformer, util

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA Pairs
qa_file = "./data/qa_pairs"  # Update with the correct file path

def load_qa_pairs(json_path):
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
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)
    
    # Find the index of the highest score
    best_match_idx = torch.argmax(cosine_scores)
    
    # Return the corresponding answer
    return answers[best_match_idx]

print("🤖 MediChat Bot: Ask me anything about nausea! Type 'exit' to stop.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🤖 Goodbye!")
        break

    response = get_answer(user_input)
    print(f"🤖 {response}")
