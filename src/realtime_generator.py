import pandas as pd
import numpy as np
import json
from collections import defaultdict, deque

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class RealtimeDataGenerator:
    def __init__(self, data_path=None, seq_len=5):
        if data_path is None:
            data_path = os.path.join(PROJECT_ROOT, 'data', 'processed_train.parquet')
            
        print(f"Loading preprocessed data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print("Data loaded successfully.")
        
        metadata_path = os.path.join(PROJECT_ROOT, 'data', 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        self.cat_cols = metadata['categorical_features']
        self.num_cols = metadata['numerical_features']
        self.cat_vocab = metadata['cat_vocab']
        self.num_cont_features = len(self.num_cols)
        self.seq_len = seq_len
        
        self.total_samples = len(self.df)
        self.current_index = 0
        
        print("Converting to numpy arrays and building padding index...")
        X_cont_raw = self.df[self.num_cols].values.astype(np.float32)
        self.X_cont = np.vstack([X_cont_raw, np.zeros(self.num_cont_features, dtype=np.float32)])
        
        self.X_cat = {}
        for col in self.cat_cols:
            cat_raw = self.df[col].values.astype(np.int32)
            self.X_cat[col] = np.append(cat_raw, 0)
            
        self.y_all = self.df['isFraud'].values.astype(np.float32)
        
        if 'card1' in self.df.columns:
            self.cards = self.df['card1'].values
        else:
            self.cards = np.zeros(self.total_samples, dtype=np.int32)
            
        self.pad_idx = self.total_samples
        self.card_hist_idx = defaultdict(lambda: deque(maxlen=self.seq_len))
        
        print("Freeing memory from pandas DataFrame...")
        del self.df
        import gc
        gc.collect()
        
        print(f"Ready to stream {self.total_samples} samples grouped by card1 (seq_len={self.seq_len}).")

        self.drift_points = []
        self._drift_offsets = []

    def set_drift(self, points, magnitude=1.0, feature_fraction=0.4, seed=123):
        """Inject abrupt covariate drift with KNOWN ground-truth locations.

        points: transaction indices at which a new shift is added (cumulative).
        Each drift adds magnitude * N(0,1) to a random subset of continuous features.
        """
        rng = np.random.RandomState(seed)
        self.drift_points = sorted(int(p) for p in points)
        self._drift_offsets = []
        for _ in self.drift_points:
            mask = (rng.rand(self.num_cont_features) < feature_fraction).astype(np.float32)
            self._drift_offsets.append((magnitude * rng.randn(self.num_cont_features) * mask).astype(np.float32))

    def _current_offset(self, start_idx):
        off = None
        for p, o in zip(self.drift_points, self._drift_offsets):
            if start_idx >= p:
                off = o if off is None else off + o
        return off

    def generate_batch(self, batch_size):
        if self.current_index + batch_size > self.total_samples:
            self.current_index = 0
            self.card_hist_idx.clear()
            
        end_idx = self.current_index + batch_size
        batch_indices = np.arange(self.current_index, end_idx)
        
        
        seq_indices = np.full((batch_size, self.seq_len), self.pad_idx, dtype=np.int32)
        
        for i, idx in enumerate(batch_indices):
            card = self.cards[idx]
            hist = self.card_hist_idx[card]
            hist.append(idx)
            
            hist_list = list(hist)
            seq_indices[i, -len(hist_list):] = hist_list
            
        batch_cont = self.X_cont[seq_indices]
        offset = self._current_offset(self.current_index)
        if offset is not None:
            batch_cont = batch_cont + offset
        
        batch_dict = {'continuous': batch_cont}
        for col in self.cat_cols:
            batch_dict[col] = self.X_cat[col][seq_indices]
            
        y_batch = self.y_all[batch_indices].reshape(-1, 1)
        
        self.current_index = end_idx
        
        return batch_dict, y_batch
        
    def reset(self):
        """Resets the generator state to begin streaming from the start."""
        self.current_index = 0
        self.card_hist_idx.clear()

if __name__ == "__main__":
    generator = RealtimeDataGenerator()
    X, y = generator.generate_batch(64)
    print("X['continuous'] shape:", X['continuous'].shape)
    print("X['ProductCD'] shape:", X['ProductCD'].shape)
    print("y shape:", y.shape)
