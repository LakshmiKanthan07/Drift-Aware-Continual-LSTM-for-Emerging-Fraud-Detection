import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional,
    BatchNormalization, LeakyReLU, LayerNormalization,
    Reshape, Multiply, GlobalAveragePooling1D, Concatenate, Flatten, Embedding, Lambda,
    Conv1D, SpatialDropout1D, MultiHeadAttention, GlobalMaxPooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


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


def build_lstm_model(seq_len, num_cont_features, cat_vocab_dict, learning_rate=0.001):
    """
    Drift-Aware Continual BiLSTM model with Categorical Embeddings.
    Takes a dictionary of inputs (continuous + one for each categorical).
    """
    inputs_dict = {}
    
    # Continuous Input
    cont_input = Input(shape=(seq_len, num_cont_features), name='continuous')
    inputs_dict['continuous'] = cont_input
    
    # Categorical Inputs & Embeddings
    embedded_features = []
    for cat_name, vocab_size in cat_vocab_dict.items():
        inp = Input(shape=(seq_len,), name=cat_name)
        inputs_dict[cat_name] = inp
        
        # Rule of thumb for embedding dimension
        embed_dim = min(50, (vocab_size + 1) // 2)
        emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, name=f'emb_{cat_name}')(inp)
        embedded_features.append(emb)
    
    if embedded_features:
        x = Concatenate(name='concat_inputs')([cont_input] + embedded_features)
    else:
        x = cont_input

    # Dynamic feature size after concatenation
    total_features = num_cont_features + sum([min(50, (v + 1) // 2) for v in cat_vocab_dict.values()])

    x = BatchNormalization(name='input_bn')(x)

    # Attention Block
    x_attn = channel_attention(x, total_features, reduction_ratio=16)
    x_attn = LayerNormalization(name='attn_ln')(x_attn)
    
    # Save the attended features for the residual skip connection (only last timestep)
    shortcut = Lambda(lambda z: z[:, -1, :], name='shortcut_last_step')(x_attn)

    # Local feature extraction
    x = Conv1D(filters=total_features, kernel_size=3, padding='same', activation='relu', name='conv1d_1')(x_attn)
    x = SpatialDropout1D(0.2, name='spatial_drop_1')(x)

    # First BiLSTM Layer
    x = Bidirectional(
        LSTM(256, return_sequences=True, kernel_regularizer=l2(1e-4)),
        name='bilstm_1'
    )(x)
    x = LayerNormalization(name='ln_1')(x)
    x = Dropout(0.2, name='drop_1')(x)

    # Second BiLSTM Layer
    x = Bidirectional(
        LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
        name='bilstm_2'
    )(x)
    x = LayerNormalization(name='ln_2')(x)
    x = Dropout(0.2, name='drop_2')(x)
    
    # Temporal Self-Attention
    x_self_attn = MultiHeadAttention(num_heads=4, key_dim=64, name='temporal_attn')(x, x)
    x = LayerNormalization(name='ln_attn')(x + x_self_attn) # Add & Norm

    # Pooling
    x_avg = GlobalAveragePooling1D(name='global_avg_pool')(x)
    x_max = GlobalMaxPooling1D(name='global_max_pool')(x)
    
    x = Concatenate(name='residual_concat')([x_avg, x_max, shortcut])

    # Classification Head
    x = Dense(128, kernel_regularizer=l2(1e-4), name='dense_1')(x)
    x = LeakyReLU(negative_slope=0.1, name='leaky_relu_1')(x)
    x = Dropout(0.2, name='drop_3')(x)
    
    x = Dense(64, kernel_regularizer=l2(1e-4), name='dense_2')(x)
    x = LeakyReLU(negative_slope=0.1, name='leaky_relu_2')(x)
    x = Dropout(0.15, name='drop_4')(x)

    outputs = Dense(1, activation='sigmoid', name='fraud_output')(x)

    # Create model mapping dictionary inputs to single output
    model = Model(inputs=inputs_dict, outputs=outputs, name='DriftAware_SE_BiLSTM_Residual')

    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=3.0, alpha=0.95),  # gamma 2→3, alpha 0.85→0.95
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model
