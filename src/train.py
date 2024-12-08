import numpy as np
from tensorflow import keras
from data_loader import load_intents
from preprocess import preprocess_data, tokenize_sentences
from model import build_model, save_model_and_tokenizer

def main():
    intents_folder = 'data/intents'
    all_intents = load_intents(intents_folder)
    training_sentences, training_labels, label_encoder = preprocess_data(all_intents)
    tokenizer, padded_sequences = tokenize_sentences(training_sentences)
    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    num_classes = len(set(training_labels))

    model = build_model(vocab_size, embedding_dim, max_len, num_classes)
    model.summary()

    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs, validation_split=0.2)

    save_model_and_tokenizer(model, tokenizer, label_encoder)

if __name__ == "__main__":
    main()