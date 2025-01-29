import os
import json

def load_intents(intents_folder):
    all_intents = []
    for filename in os.listdir(intents_folder):
        if filename.endswith('.json'):
            with open(os.path.join(intents_folder, filename)) as file:
                data = json.load(file)
                all_intents.append(data)
    return all_intents

def load_patterns_and_tags(folder_path):
    patterns = []
    tags = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
                for intent in data['intents']:
                    patterns.extend(intent['patterns'])
                    tags.extend([intent['tag']] * len(intent['patterns']))
    return patterns, tags