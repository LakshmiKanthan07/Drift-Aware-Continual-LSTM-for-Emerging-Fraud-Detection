import numpy as np
import random

class StreamingDataGenerator:
    def __init__(self, num_features=10, seq_len=5, fraud_ratio=0.05):
        self.num_features = num_features
        self.seq_len = seq_len
        self.fraud_ratio = fraud_ratio
        
        # Base distributions
        self.normal_mean = np.zeros(num_features)
        self.normal_cov = np.eye(num_features)
        
        # Fraud distribution state
        self.current_fraud_concept = 0
        self._set_fraud_distribution()

    def _set_fraud_distribution(self):
        """Defines the distribution for fraudulent transactions based on the current concept."""
        if self.current_fraud_concept == 0:
            # Concept 0: High values in first few features
            self.fraud_mean = np.ones(self.num_features) * 2.5
            self.fraud_mean[self.num_features//2:] = 0
            self.fraud_cov = np.eye(self.num_features)
        elif self.current_fraud_concept == 1:
            # Concept 1: Negative values in last few features
            self.fraud_mean = np.zeros(self.num_features)
            self.fraud_mean[self.num_features//2:] = -3.0
            self.fraud_cov = np.eye(self.num_features) * 1.5
        elif self.current_fraud_concept == 2:
            # Concept 2: High variance across all features (hiding in noise)
            self.fraud_mean = np.zeros(self.num_features)
            self.fraud_cov = np.eye(self.num_features) * 4.0
        else:
            # Fallback random concept
            self.fraud_mean = np.random.uniform(-3, 3, self.num_features)
            self.fraud_cov = np.eye(self.num_features) * np.random.uniform(0.5, 2.0)

    def trigger_drift(self):
        """Manually triggers a concept drift to a new fraud pattern."""
        self.current_fraud_concept += 1
        self._set_fraud_distribution()
        print(f"--- CONCEPT DRIFT TRIGGERED: Switched to fraud concept {self.current_fraud_concept} ---")

    def generate_batch(self, batch_size):
        """Generates a batch of sequences for the LSTM. Shape: (batch_size, seq_len, num_features)"""
        X_batch = np.zeros((batch_size, self.seq_len, self.num_features))
        y_batch = np.zeros((batch_size, 1))

        for i in range(batch_size):
            is_fraud = random.random() < self.fraud_ratio
            y_batch[i, 0] = 1.0 if is_fraud else 0.0

            for t in range(self.seq_len):
                if is_fraud and t == self.seq_len - 1:
                    # The actual fraudulent behavior manifests at the end of the sequence
                    features = np.random.multivariate_normal(self.fraud_mean, self.fraud_cov)
                else:
                    # Normal transaction behavior
                    features = np.random.multivariate_normal(self.normal_mean, self.normal_cov)
                
                X_batch[i, t, :] = features

        return X_batch, y_batch

if __name__ == "__main__":
    # Test generator
    generator = StreamingDataGenerator()
    X, y = generator.generate_batch(32)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Number of frauds in batch: {np.sum(y)}")
    generator.trigger_drift()
    X2, y2 = generator.generate_batch(32)
