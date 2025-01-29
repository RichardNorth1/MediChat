from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
import os

# Load the JSON dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "escalation_dataset.json")
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to Hugging Face Dataset format
data_dict = {"text": [entry["text"] for entry in data], "label": [entry["label"] for entry in data]}
dataset = Dataset.from_dict(data_dict)

# Split into train and validation sets
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Load a pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply tokenization to both train and validation datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)  # Fix here

# Set format for PyTorch
tokenized_train = tokenized_train.with_format("torch")
tokenized_val = tokenized_val.with_format("torch")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

# Define Trainer
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "fine_tuned_model")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("Model training complete and saved.")
