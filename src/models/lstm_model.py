import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam


def build_lstm_model(seq_len, num_features, learning_rate=0.001):

    model = Sequential([
        Input(shape=(seq_len, num_features)),

        # First BiLSTM layer – returns full sequence for the second layer
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),

        # Second BiLSTM layer – condenses to a single context vector
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid'),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


if __name__ == "__main__":
    model = build_lstm_model(seq_len=5, num_features=10)
    model.summary()
