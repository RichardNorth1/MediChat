import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

# Load JSON Data
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["qa_pairs"]

# Prepare the Data
def prepare_data(qa_pairs):
    data = {
        "text": [pair["question"] for pair in qa_pairs],
        "label": [pair["answer"] for pair in qa_pairs]
    }
    return Dataset.from_dict(data)

# Paths
json_file_path = "./data/nausea_qa_pairs.json"

# Load Data
qa_pairs = load_data(json_file_path)

# Prepare Dataset
dataset = prepare_data(qa_pairs)

# Split Dataset into Train and Validation
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Load Tokenizer and Pretrained Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize Data
def preprocess_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)

# Ensure the label column is renamed correctly
train_dataset = train_dataset.remove_columns(["text"]).rename_column("label", "labels")
val_dataset = val_dataset.remove_columns(["text"]).rename_column("label", "labels")

# Set the dataset format
train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Calculate number of labels
num_labels = len(set(dataset["train"]["label"]))

# Load Pretrained Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./qa_model",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the Model
trainer.train()

# Save the Model
model_path = "./data/models/qa_model"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("Training complete and model saved.")
