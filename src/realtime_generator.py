import pandas as pd
import numpy as np
import json
from collections import defaultdict, deque

class RealtimeDataGenerator:
    def __init__(self, data_path='data/processed_train.parquet', seq_len=5):
        print(f"Loading preprocessed data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print("Data loaded successfully.")
        
        with open('data/metadata.json', 'r') as f:
            metadata = json.load(f)
            
        self.cat_cols = metadata['categorical_features']
        self.num_cols = metadata['numerical_features']
        self.cat_vocab = metadata['cat_vocab']
        self.num_cont_features = len(self.num_cols)
        self.seq_len = seq_len
        
        self.total_samples = len(self.df)
        self.current_index = 0
        
        print("Converting to numpy arrays and building padding index...")
        # Create continuous array and append a zero row for padding at the very end
        X_cont_raw = self.df[self.num_cols].values.astype(np.float32)
        self.X_cont = np.vstack([X_cont_raw, np.zeros(self.num_cont_features, dtype=np.float32)])
        
        # Create dict of categorical arrays and append a zero for padding
        self.X_cat = {}
        for col in self.cat_cols:
            cat_raw = self.df[col].values.astype(np.int32)
            self.X_cat[col] = np.append(cat_raw, 0)
            
        self.y_all = self.df['isFraud'].values.astype(np.float32)
        
        # Use card1 as the unique grouping identifier
        if 'card1' in self.df.columns:
            self.cards = self.df['card1'].values
        else:
            # Fallback if card1 is somehow not there (it is in our data)
            self.cards = np.zeros(self.total_samples, dtype=np.int32)
            
        self.pad_idx = self.total_samples
        self.card_hist_idx = defaultdict(lambda: deque(maxlen=self.seq_len))
        
        print(f"Ready to stream {self.total_samples} samples grouped by card1 (seq_len={self.seq_len}).")

    def generate_batch(self, batch_size):
        if self.current_index + batch_size > self.total_samples:
            self.current_index = 0
            self.card_hist_idx.clear()
            
        end_idx = self.current_index + batch_size
        batch_indices = np.arange(self.current_index, end_idx)
        
        # Build sequence indices shape (batch_size, seq_len)
        seq_indices = np.full((batch_size, self.seq_len), self.pad_idx, dtype=np.int32)
        
        for i, idx in enumerate(batch_indices):
            card = self.cards[idx]
            hist = self.card_hist_idx[card]
            hist.append(idx)
            
            hist_list = list(hist)
            seq_indices[i, -len(hist_list):] = hist_list
            
        # Fetch data using advanced indexing
        batch_cont = self.X_cont[seq_indices]
        
        batch_dict = {'continuous': batch_cont}
        for col in self.cat_cols:
            batch_dict[col] = self.X_cat[col][seq_indices]
            
        # Labels are just the label of the *current* transaction (the last item in the sequence)
        y_batch = self.y_all[batch_indices].reshape(-1, 1)
        
        self.current_index = end_idx
        
        return batch_dict, y_batch

if __name__ == "__main__":
    generator = RealtimeDataGenerator()
    X, y = generator.generate_batch(64)
    print("X['continuous'] shape:", X['continuous'].shape)
    print("X['ProductCD'] shape:", X['ProductCD'].shape)
    print("y shape:", y.shape)
