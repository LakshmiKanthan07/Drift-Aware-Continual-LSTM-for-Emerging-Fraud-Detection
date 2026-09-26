import random
import numpy as np

class RandomReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, x_dict, y):
        """
        Adds a single sample to the buffer.
        x_dict: dictionary of arrays for a single sample
        y: label array for a single sample
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append((x_dict, y))
        else:
            idx = random.randint(0, self.capacity - 1)
            self.buffer[idx] = (x_dict, y)

    def sample(self, batch_size):
        """Samples a mini-batch from the buffer. Output X is a dict of arrays."""
        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)
            
        y_batch = np.array([item[1] for item in batch])
        
        # Build dict of arrays
        X_batch = {}
        if batch:
            keys = batch[0][0].keys()
            for k in keys:
                X_batch[k] = np.array([item[0][k] for item in batch])
                
        return X_batch, y_batch

    def clear(self):
        self.buffer = []

    def partial_clear(self, clear_fraud=False):
        """Compatibility shim — clears the whole buffer (no class separation)."""
        self.buffer = []

    @property
    def size(self):
        return len(self.buffer)
