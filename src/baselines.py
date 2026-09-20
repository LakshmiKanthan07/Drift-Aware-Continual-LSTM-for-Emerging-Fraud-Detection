import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, cohen_kappa_score, average_precision_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LeakyReLU, Concatenate, Flatten, Embedding
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import json

from continual_learner import AdaptiveThreshold

class StaticMLPBaseline:
    def __init__(self, seq_len, cat_vocab_dict, num_cont_features, learning_rate=0.001):
        self.model_type = 'Static MLP Baseline'
        self.seq_len = seq_len
        self.cat_vocab_dict = cat_vocab_dict
        self.num_cont_features = num_cont_features
        
        self.model = self._build_mlp_model(learning_rate)
        self.adaptive_threshold = AdaptiveThreshold(window_size=1000)
        self.metrics_log = []

    def _build_mlp_model(self, learning_rate):
        inputs_dict = {}
        
        # Continuous Input
        cont_input = Input(shape=(self.seq_len, self.num_cont_features), name='continuous')
        inputs_dict['continuous'] = cont_input
        
        # Categorical Inputs & Embeddings
        embedded_features = []
        for cat_name, vocab_size in self.cat_vocab_dict.items():
            inp = Input(shape=(self.seq_len,), name=cat_name)
            inputs_dict[cat_name] = inp
            
            embed_dim = min(50, (vocab_size + 1) // 2)
            emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, name=f'emb_{cat_name}')(inp)
            embedded_features.append(emb)
        
        if embedded_features:
            x = Concatenate(name='concat_inputs')([cont_input] + embedded_features)
        else:
            x = cont_input

        # Flatten out the sequence dimension (treat as standard MLP)
        x = Flatten()(x)

        x = BatchNormalization()(x)
        
        x = Dense(256, kernel_regularizer=l2(1e-4))(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.4)(x)
        
        x = Dense(128, kernel_regularizer=l2(1e-4))(x)
        x = LeakyReLU(negative_slope=0.1)(x)
        x = Dropout(0.3)(x)
        
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs_dict, outputs=outputs, name='Static_MLP')
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _batch_metrics(self, preds_binary, preds_prob, y):
        acc = float(np.mean(preds_binary == y))

        tp = int(np.sum((preds_binary == 1) & (y == 1)))
        fp = int(np.sum((preds_binary == 1) & (y == 0)))
        fn = int(np.sum((preds_binary == 0) & (y == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        try:
            roc_auc = float(roc_auc_score(y, preds_prob))
        except ValueError:
            roc_auc = 0.5  

        try:
            pr_auc = float(average_precision_score(y, preds_prob))
        except ValueError:
            pr_auc = 0.0  

        try:
            kappa = float(cohen_kappa_score(y, preds_binary))
        except Exception:
            kappa = 0.0

        return acc, precision, recall, f1, roc_auc, pr_auc, kappa

    def process_stream(self, generator, num_batches, batch_size=64):
        print(f"\nStarting simulation for baseline: {self.model_type}")

        # Bootstrap Phase
        print("Bootstrapping static MLP on first 250 batches...")
        X_bootstrap_dict = {col: [] for col in generator.cat_cols + ['continuous']}
        y_bootstrap = []
        
        for _ in range(250):
            X_dict, y = generator.generate_batch(batch_size)
            for col in X_dict:
                X_bootstrap_dict[col].append(X_dict[col])
            y_bootstrap.append(y)

        # Concatenate for training
        X_train = {col: np.vstack(X_bootstrap_dict[col]) for col in X_bootstrap_dict}
        y_train = np.vstack(y_bootstrap)

        print(f"Training Static MLP on {len(y_train)} samples for 5 epochs...")
        self.model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1,
                       class_weight={0: 1.0, 1: 29})  # ~1/0.035 fraud upweight
        
        # Free memory from bootstrap phase
        del X_bootstrap_dict
        del y_bootstrap
        del X_train
        del y_train
        import gc
        gc.collect()

        # Streaming Phase
        drifts_detected = []
        
        for batch_idx in tqdm(range(num_batches), desc=self.model_type):
            X_dict, y = generator.generate_batch(batch_size)
            
            preds_prob = self.model(X_dict, training=False).numpy().ravel()
            y = y.ravel()

            self.adaptive_threshold.add(y, preds_prob)
            if batch_idx > 0 and batch_idx % 100 == 0:
                self.adaptive_threshold.update_threshold()
            threshold = self.adaptive_threshold.current_threshold
            
            preds_binary = (preds_prob >= threshold).astype(int)

            acc, prec, rec, f1, roc_auc, pr_auc, kappa = self._batch_metrics(preds_binary, preds_prob, y)

            self.metrics_log.append({
                'batch': batch_idx,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'kappa': kappa,
                'threshold': threshold
            })

        safe_name = self.model_type.replace(' ', '_')
        df_metrics = pd.DataFrame(self.metrics_log)
        df_metrics.to_csv(f'data/{safe_name}_metrics.csv', index=False)
        pd.DataFrame(drifts_detected).to_csv(f'data/{safe_name}_drifts.csv', index=False)

        print(f"Finished {self.model_type}. Saved to data/")
        return df_metrics, drifts_detected
