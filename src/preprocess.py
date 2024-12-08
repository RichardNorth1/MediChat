import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_data(all_intents):
    training_sentences = []
    training_labels = []
    responses = []

    for data in all_intents:
        for intent in data['intents']:
            for pattern in intent['patterns']:
                training_sentences.append(pattern)
                training_labels.append(intent['tag'])
            responses.append(intent['responses'])

    label_encoder = LabelEncoder()
    training_labels = label_encoder.fit_transform(training_labels)

    return training_sentences, training_labels, label_encoder

def tokenize_sentences(training_sentences, vocab_size=1000, oov_token="<OOV>", max_len=20):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)
    return tokenizer, padded_sequences