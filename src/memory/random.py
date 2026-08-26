import random
import numpy as np

class RandomReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, x, y):
        """
        Adds a single sample to the buffer.
        If buffer is full, replaces a random existing sample (Reservoir sampling idea).
        x shape: (seq_len, num_features)
        y shape: (1,)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            # Randomly replace an old sample
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = (x, y)

    def sample(self, batch_size):
        """Samples a mini-batch from the buffer."""
        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)
            
        X_batch = np.array([item[0] for item in batch])
        y_batch = np.array([item[1] for item in batch])
        return X_batch, y_batch

    def __len__(self):
        return len(self.buffer)
