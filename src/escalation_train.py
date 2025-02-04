import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Load the 'all-MiniLM-L6-v2' model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 2: Example conversations (You can replace this with your dataset)
data_path = "./data/escalation_dataset.json"    
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# Step 3: Extract the sentences and labels
sentences = [item[0] for item in data]
labels = [item[1] for item in data]

# Step 4: Convert sentences to embeddings
embeddings = model.encode(sentences)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Step 6: Train a classifier (Logistic Regression in this case)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Step 7: Evaluate the classifier
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 8: Use the model to predict escalation for new input
def predict_escalation(user_input):
    embedding = model.encode([user_input])
    prediction = classifier.predict(embedding)
    if prediction == 1:
        return "Escalate the conversation to a human."
    else:
        return "The conversation can be handled by the bot."

# Example usage
user_input = "I need someone to help me immediately!"
result = predict_escalation(user_input)
print(result)
