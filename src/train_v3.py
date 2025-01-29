import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Load and prepare the QA dataset
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["qa_pairs"]  # Expecting a format with "question" and "answer"

# Encode the answers
def encode_labels(answers):
    # The output (answer) is already text, so no need for label encoding
    return answers

# Paths
qa_file_path = "./data/nausea_qa_pairs.json"  # Make sure this is the correct path to your JSON file
qa_pairs = load_data(qa_file_path)

# Separate questions and answers
questions = [pair["question"] for pair in qa_pairs]
answers = [pair["answer"] for pair in qa_pairs]

# Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(questions, answers, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"question": train_texts, "answer": train_labels})
val_dataset = Dataset.from_dict({"question": val_texts, "answer": val_labels})

# Load the pretrained tokenizer
model_name = "t5-small"  # You can use "t5-base" or "t5-large" for larger models
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data
def preprocess_data(example):
    input_texts = [f"question: {q}" for q in example["question"]]
    target_texts = [a for a in example["answer"]]

    model_inputs = tokenizer(input_texts, truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(target_texts, truncation=True, padding="max_length", max_length=64)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["question", "answer"])
val_dataset = val_dataset.remove_columns(["question", "answer"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Load the pretrained seq2seq model (T5)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "qa_model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save label mapping (not really needed for QA, but we can save it anyway)
label_map = {"qa_pairs": qa_pairs}
with open(model_path + "/label_mapping.json", "w") as f:
    json.dump(label_map, f)

print("Model training completed and saved.")
