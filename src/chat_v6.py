import os
import json
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import colorama
colorama.init()
from colorama import Fore, Style
import language_tool_python

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load QA Pairs
qa_file = "./data/qa_pairs"

# Load Sentence Transformer model for the chatbot
chatbot_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Load Sentence Transformer model for the escalation feature
escalation_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

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

# Load the data for the classifier
escalation_data = "./data/escalation_dataset.json"
with open(escalation_data, "r") as file:
    data = json.load(file)

# Extract the sentences and labels
sentences = [item['text'] for item in data]
labels = [item['label'] for item in data]

# Convert stored questions to embeddings
stored_embeddings = chatbot_model.encode(questions, convert_to_tensor=True, device=device)

# Convert sentences to embeddings
embeddings = chatbot_model.encode(sentences, device=device)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# train a logistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)

# check the model performance
accuracy = logistic_regression_model.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

def get_answer(query):
    """Get the best answer for the query."""
    query_embedding = chatbot_model.encode(query, convert_to_tensor=True, device=device)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings.to(device))
    
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

def predict_escalation(user_input):
    """Predict whether the input text requires escalation."""
    embedding = escalation_model.encode([user_input])
    prediction = logistic_regression_model.predict(embedding)
    if prediction == 1:
        return True
    else:
        return False

def handle_chat(input_text):
    """Handle the chat by predicting the intent and providing a response."""
    # Correct grammar in the input text
    corrected_inp = correct_grammar(input_text)

    if predict_escalation(corrected_inp):
        return {"message": "Escalating the conversation to a human.", "escalate": True}
    
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