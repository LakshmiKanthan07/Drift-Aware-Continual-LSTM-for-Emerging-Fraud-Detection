import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional,
    BatchNormalization, LeakyReLU, LayerNormalization,
    Reshape, Multiply, GlobalAveragePooling1D, Concatenate, Flatten
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
    """
    squeezed = GlobalAveragePooling1D()(x)   # (batch, num_features)
    r = max(1, num_features // reduction_ratio)
    excited = Dense(r, activation='relu', use_bias=False,
                    kernel_regularizer=l2(1e-4), name='se_fc1')(squeezed)
    excited = Dense(num_features, activation='sigmoid', use_bias=False,
                    kernel_regularizer=l2(1e-4), name='se_fc2')(excited)
    excited = Reshape((1, num_features), name='se_reshape')(excited)
    return Multiply(name='se_scale')([x, excited])


def build_lstm_model(seq_len, num_features, learning_rate=0.001):
    """
    Drift-Aware Continual BiLSTM model with:
    - Squeeze-and-Excitation feature attention
    - Residual Skip Connection bypassing BiLSTMs to improve gradient flow during adaptation
    - Two Bidirectional LSTM layers (128, 64) with L2 regularization
    - Focal Loss (gamma=2.0, alpha=0.50) for extreme class imbalance
    """
    inputs = Input(shape=(seq_len, num_features), name='transaction_input')

    x = BatchNormalization(name='input_bn')(inputs)

    # Attention Block
    x_attn = channel_attention(x, num_features, reduction_ratio=16)
    x_attn = LayerNormalization(name='attn_ln')(x_attn)
    
    # Save the attended features for the residual skip connection
    shortcut = Flatten(name='shortcut_flatten')(x_attn)

    # First BiLSTM Layer
    x = Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        name='bilstm_1'
    )(x_attn)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.4, name='drop_1')(x)

    # Second BiLSTM Layer
    x = Bidirectional(
        LSTM(64, return_sequences=False, kernel_regularizer=l2(1e-4)),
        name='bilstm_2'
    )(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.4, name='drop_2')(x)
    
    # -----------------------------------------------------------------------
    # RESIDUAL SKIP CONNECTION
    # Concatenate the original attended features directly with the BiLSTM output.
    # This acts like a ResNet shortcut, dramatically improving gradient flow
    # to the attention weights and feature layers during rapid 10-epoch drift tuning.
    # -----------------------------------------------------------------------
    x = Concatenate(name='residual_concat')([x, shortcut])

    # Classification Head
    x = Dense(64, kernel_regularizer=l2(1e-4), name='dense_1')(x)
    x = LeakyReLU(negative_slope=0.1, name='leaky_relu')(x)
    x = Dropout(0.3, name='drop_3')(x)

    outputs = Dense(1, activation='sigmoid', name='fraud_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='DriftAware_SE_BiLSTM_Residual')

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
