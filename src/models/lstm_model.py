import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_lstm_model(seq_len, num_features, learning_rate=0.001):
    """
    Builds and compiles an LSTM model for binary fraud classification.

    Uses a proper Input layer (replaces deprecated input_shape on LSTM)
    to silence Keras warnings. Compiled with standard binary cross-entropy;
    class weighting is applied at training time in the learner.
    """
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
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
