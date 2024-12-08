from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
import pickle

def build_model(vocab_size, embedding_dim=16, max_len=20, num_classes=10):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy'])
    return model

def save_model_and_tokenizer(model, tokenizer, label_encoder, model_path="data/models/chat_model.h5", tokenizer_path="data/models/tokenizer.pickle", label_encoder_path="data/models/label_encoder.pickle"):
    model.save(model_path)
    with open(tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(label_encoder_path, 'wb') as enc:
        pickle.dump(label_encoder, enc, protocol=pickle.HIGHEST_PROTOCOL)