import pandas as pd
import numpy as np

class RealtimeDataGenerator:
    def __init__(self, data_path='data/processed_train.parquet'):
        print(f"Loading preprocessed data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print("Data loaded successfully.")
        
        # Identify feature columns and label
        self.feature_cols = [c for c in self.df.columns if c not in ['isFraud', 'TransactionID', 'TransactionDT']]
        self.num_features = len(self.feature_cols)
        self.seq_len = 1
        
        self.current_index = 0
        self.total_samples = len(self.df)
        
        # Pre-convert to numpy arrays for much faster batch slicing
        print("Converting to numpy arrays for fast streaming...")
        self.X_all = self.df[self.feature_cols].values
        self.y_all = self.df['isFraud'].values
        print(f"Ready to stream {self.total_samples} samples with {self.num_features} features.")

    def generate_batch(self, batch_size):
        """Generates a batch sequentially. Shape: (batch_size, seq_len=1, num_features)"""
        if self.current_index + batch_size > self.total_samples:
            # If we run out of data, just wrap around (or we could stop)
            # In a real stream, we'd wait for more data. For simulation, wrap around.
            self.current_index = 0
            
        end_idx = self.current_index + batch_size
        
        X_batch_2d = self.X_all[self.current_index:end_idx]
        y_batch_1d = self.y_all[self.current_index:end_idx]
        
        # Reshape to (batch_size, seq_len, num_features)
        X_batch = X_batch_2d.reshape((batch_size, 1, self.num_features))
        y_batch = y_batch_1d.reshape((batch_size, 1))
        
        self.current_index = end_idx
        
        return X_batch, y_batch

if __name__ == "__main__":
    # Test generator
    generator = RealtimeDataGenerator()
    X, y = generator.generate_batch(64)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Number of frauds in batch: {np.sum(y)}")
