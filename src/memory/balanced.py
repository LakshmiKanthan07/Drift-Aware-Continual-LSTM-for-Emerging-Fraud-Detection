import random
import numpy as np

class BalancedReplayBuffer:
    def __init__(self, capacity=5000):
        """
        Maintains separate buffers for normal and fraud transactions to ensure
        balanced sampling during replay.
        """
        self.capacity = capacity
        self.half_capacity = capacity // 2
        
        self.normal_buffer = []
        self.fraud_buffer = []
        
    def add(self, x_dict, y):
        """
        Adds a single sample to the appropriate buffer.
        """
        is_fraud = (y[0] == 1.0)
        buffer = self.fraud_buffer if is_fraud else self.normal_buffer
        
        if len(buffer) < self.half_capacity:
            buffer.append((x_dict, y))
        else:
            # Randomly replace an old sample
            idx = random.randint(0, self.half_capacity - 1)
            buffer[idx] = (x_dict, y)

    def sample(self, batch_size):
        """
        Samples a balanced mini-batch. Output X is a dict of arrays.
        """
        half_batch = batch_size // 2
        
        # Sample normal
        n_sample = min(len(self.normal_buffer), half_batch)
        normals = random.sample(self.normal_buffer, n_sample) if n_sample > 0 else []
        
        # Sample fraud
        f_sample = min(len(self.fraud_buffer), batch_size - n_sample)
        frauds = random.sample(self.fraud_buffer, f_sample) if f_sample > 0 else []
        
        # In case we couldn't get enough frauds, pad with normals
        remaining = batch_size - (len(normals) + len(frauds))
        if remaining > 0 and len(self.normal_buffer) > len(normals):
            extra_n = min(remaining, len(self.normal_buffer) - len(normals))
            # Just sample randomly for the remainder
            normals.extend(random.sample(self.normal_buffer, extra_n))

        batch = normals + frauds
        random.shuffle(batch)
        
        if not batch:
            return {}, np.array([])
            
        y_batch = np.array([item[1] for item in batch])
        
        # Build dict of arrays
        X_batch = {}
        keys = batch[0][0].keys()
        for k in keys:
            X_batch[k] = np.array([item[0][k] for item in batch])
            
        return X_batch, y_batch

    def clear(self):
        """Clears all samples from the buffer."""
        self.normal_buffer = []
        self.fraud_buffer = []

    def partial_clear(self, clear_fraud=False):
        """
        Clears the normal buffer to forget old concepts upon drift.
        Optionally keeps the fraud buffer intact to preserve rare signals.
        """
        self.normal_buffer = []
        if clear_fraud:
            self.fraud_buffer = []

    @property
    def size(self):
        return len(self.normal_buffer) + len(self.fraud_buffer)
