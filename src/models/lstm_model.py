import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional,
    BatchNormalization, LeakyReLU, LayerNormalization,
    Reshape, Permute, Activation, Multiply, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K


def focal_loss(gamma=2.0, alpha=0.50):
    """
    Focal Loss with alpha=0.50 to aggressively up-weight the rare fraud class.
    gamma=2.0 focuses learning on hard, misclassified examples.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = K.pow(1.0 - p_t, gamma)
        return K.mean(alpha_factor * modulating_factor * cross_entropy, axis=-1)
    return focal_loss_fixed


def channel_attention(x, num_features, reduction_ratio=16):
    """
    Squeeze-and-Excitation style channel attention over the feature dimension.
    Since seq_len=1 in this tabular streaming setup, we treat each of the
    431 features as a 'channel' and learn which features matter most for fraud.
    
    This is the correct attention mechanism for (batch, 1, features) input shapes.
    """
    # Squeeze: average over the time dimension → (batch, features)
    squeezed = GlobalAveragePooling1D()(x)   # (batch, num_features)

    # Excitation: two-layer MLP to learn feature importance weights
    r = max(1, num_features // reduction_ratio)
    excited = Dense(r, activation='relu', use_bias=False,
                    kernel_regularizer=l2(1e-4), name='se_fc1')(squeezed)
    excited = Dense(num_features, activation='sigmoid', use_bias=False,
                    kernel_regularizer=l2(1e-4), name='se_fc2')(excited)   # (batch, num_features)

    # Reshape for broadcasting: (batch, 1, num_features)
    excited = Reshape((1, num_features), name='se_reshape')(excited)

    # Scale: element-wise multiply original input by importance weights
    return Multiply(name='se_scale')([x, excited])


def build_lstm_model(seq_len, num_features, learning_rate=0.001):
    """
    Drift-Aware Continual BiLSTM model with:
    - Squeeze-and-Excitation feature attention (works correctly for seq_len=1)
    - Two Bidirectional LSTM layers (128, 64) with L2 regularization
    - Focal Loss (gamma=2.0, alpha=0.50) for extreme class imbalance
    """
    inputs = Input(shape=(seq_len, num_features), name='transaction_input')

    # -----------------------------------------------------------------------
    # Input normalization
    # -----------------------------------------------------------------------
    x = BatchNormalization(name='input_bn')(inputs)

    # -----------------------------------------------------------------------
    # Squeeze-and-Excitation Feature Attention Block
    # Learns which of the 431 features are most predictive of fraud.
    # Works correctly even when seq_len=1 (tabular streaming data).
    # -----------------------------------------------------------------------
    x = channel_attention(x, num_features, reduction_ratio=16)
    x = LayerNormalization(name='attn_ln')(x)

    # -----------------------------------------------------------------------
    # First BiLSTM Layer
    # -----------------------------------------------------------------------
    x = Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        name='bilstm_1'
    )(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.4, name='drop_1')(x)

    # -----------------------------------------------------------------------
    # Second BiLSTM Layer – condenses to a single context vector
    # -----------------------------------------------------------------------
    x = Bidirectional(
        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)),
        name='bilstm_2'
    )(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.4, name='drop_2')(x)

    # -----------------------------------------------------------------------
    # Classification Head
    # -----------------------------------------------------------------------
    x = Dense(64, kernel_regularizer=l2(1e-4), name='dense_1')(x)
    x = LeakyReLU(negative_slope=0.1, name='leaky_relu')(x)
    x = Dropout(0.3, name='drop_3')(x)

    outputs = Dense(1, activation='sigmoid', name='fraud_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='DriftAware_SE_BiLSTM')

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.50),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


if __name__ == "__main__":
    model = build_lstm_model(seq_len=1, num_features=431)
    model.summary()
