import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
from src.data_loader import load_patterns_and_tags

# Encode the labels (convert tags to numerical values)
def encode_labels(labels):
    unique_labels = list(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    encoded_labels = [label2id[label] for label in labels]
    return encoded_labels, label2id, id2label

# Paths
intents_folder = "./data/intents"
patterns, tags = load_patterns_and_tags(intents_folder)

# Encode labels
encoded_tags, label2id, id2label = encode_labels(tags)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(patterns, encoded_tags, test_size=0.2, random_state=42)

# Convert data to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

# Tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_data(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_data, batched=True)
val_dataset = val_dataset.map(preprocess_data, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Load Pretrained Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {
        "accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()
    }
)

# train model
trainer.train()

# Save Model
model_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "intent_model")

model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

# Save Label Mapping
with open(model_path + "/label_mapping.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

print("Model training completed and saved.")
