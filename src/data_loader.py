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