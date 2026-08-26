import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from data_generator import StreamingDataGenerator
from models.lstm_model import build_lstm_model
from drift_detection.adwin import ADWINDriftDetector
from drift_detection.window import WindowDriftDetector
from memory.random import RandomReplayBuffer
from memory.balanced import BalancedReplayBuffer

class ContinualLearner:
    def __init__(self, config_name, detector_type, memory_type, seq_len=5, num_features=10):
        self.config_name = config_name
        self.model = build_lstm_model(seq_len, num_features)
        
        if detector_type == 'adwin':
            self.detector = ADWINDriftDetector()
        elif detector_type == 'window':
            self.detector = WindowDriftDetector()
        else:
            self.detector = None # Static baseline
            
        if memory_type == 'random':
            self.memory = RandomReplayBuffer()
        elif memory_type == 'balanced':
            self.memory = BalancedReplayBuffer()
        else:
            self.memory = None # No memory

        self.metrics_log = []
        self.drift_events = []

    def process_stream(self, generator, num_batches, batch_size=32, drift_prob=0.005):
        """Processes a data stream, simulating continual learning."""
        
        print(f"Starting simulation for configuration: {self.config_name}")
        
        # Initial training on first 50 batches to bootstrap the model
        print("Bootstrapping model...")
        for _ in range(50):
            X, y = generator.generate_batch(batch_size)
            self.model.train_on_batch(X, y)
            if self.memory:
                for i in range(batch_size):
                    self.memory.add(X[i], y[i])
                    
        # Streaming phase
        for batch_idx in tqdm(range(num_batches)):
            # Randomly trigger concept drift based on probability
            if random.random() < drift_prob and batch_idx > 100:
                generator.trigger_drift()
                
            X, y = generator.generate_batch(batch_size)
            
            # Predict
            preds = self.model.predict(X, verbose=0)
            preds_binary = (preds > 0.5).astype(int)
            
            # Calculate batch metrics
            acc = np.mean(preds_binary == y)
            
            tp = np.sum((preds_binary == 1) & (y == 1))
            fp = np.sum((preds_binary == 1) & (y == 0))
            fn = np.sum((preds_binary == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.metrics_log.append({
                'batch': batch_idx,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })

            # Add to memory
            if self.memory:
                for i in range(batch_size):
                    self.memory.add(X[i], y[i])

            # Check for drift
            drift_detected = False
            if self.detector:
                if isinstance(self.detector, ADWINDriftDetector):
                    # Feed errors to ADWIN (0 if correct, 1 if wrong)
                    for i in range(batch_size):
                        error = 0 if preds_binary[i] == y[i] else 1
                        if self.detector.update(error):
                            drift_detected = True
                            self.detector.reset()
                elif isinstance(self.detector, WindowDriftDetector):
                    # Feed correctness to Window Metric
                    for i in range(batch_size):
                        is_correct = 1 if preds_binary[i] == y[i] else 0
                        if self.detector.update(is_correct):
                            drift_detected = True
                            self.detector.reset()

            # Adaptation phase
            if drift_detected:
                print(f"\n[!] Drift detected at batch {batch_idx}! Adapting model...")
                self.drift_events.append(batch_idx)
                
                # Fine-tune the model using memory and current batch
                if self.memory:
                    # Train for a few epochs on a mix of new data and memory
                    for _ in range(3): # Fine-tuning steps
                        X_mem, y_mem = self.memory.sample(batch_size * 2)
                        if len(X_mem) > 0:
                            # Combine memory and current batch
                            X_train = np.vstack((X_mem, X))
                            y_train = np.vstack((y_mem, y))
                            self.model.train_on_batch(X_train, y_train)
                else:
                    # Naive fine-tuning (only on new data, prone to forgetting)
                    for _ in range(3):
                        self.model.train_on_batch(X, y)

        return pd.DataFrame(self.metrics_log), self.drift_events

def run_simulation():
    # Simulation Parameters
    NUM_TRANSACTIONS = 500000
    BATCH_SIZE = 64
    NUM_BATCHES = NUM_TRANSACTIONS // BATCH_SIZE
    DRIFT_PROB = 0.0005 # Probability of drift per batch
    
    # Save directory
    os.makedirs('data', exist_ok=True)
    
    configs = [
        {'name': 'Static Baseline', 'detector': None, 'memory': None},
        {'name': 'Config A (ADWIN + Random)', 'detector': 'adwin', 'memory': 'random'},
        {'name': 'Config B (Window + Balanced)', 'detector': 'window', 'memory': 'balanced'}
    ]
    
    for conf in configs:
        generator = StreamingDataGenerator(fraud_ratio=0.03) # 3% fraud
        learner = ContinualLearner(
            config_name=conf['name'], 
            detector_type=conf['detector'], 
            memory_type=conf['memory']
        )
        
        df, drifts = learner.process_stream(generator, num_batches=NUM_BATCHES, batch_size=BATCH_SIZE, drift_prob=DRIFT_PROB)
        
        # Save results
        safe_name = conf['name'].replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'and')
        df.to_csv(f"data/{safe_name}_metrics.csv", index=False)
        pd.DataFrame({'drift_batch': drifts}).to_csv(f"data/{safe_name}_drifts.csv", index=False)
        
        print(f"Finished {conf['name']}. Saved to data/\n")

if __name__ == "__main__":
    run_simulation()
