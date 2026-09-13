import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_ieee_data():
    print("Loading data...")
    # Load train data
    train_transaction = pd.read_csv('data/train_transaction.csv')
    train_identity = pd.read_csv('data/train_identity.csv')

    print(f"Transaction shape: {train_transaction.shape}")
    print(f"Identity shape: {train_identity.shape}")

    # Merge on TransactionID
    print("Merging data...")
    train = train_transaction.merge(train_identity, on='TransactionID', how='left')

    del train_transaction, train_identity
    gc.collect()

    print(f"Merged train shape: {train.shape}")

    # Sort by time (essential for data stream simulation)
    train = train.sort_values('TransactionDT').reset_index(drop=True)

    # Identify categorical columns based on the Kaggle data description
    categorical_features = ['ProductCD'] + \
                           [f'card{i}' for i in range(1, 7)] + \
                           ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + \
                           [f'M{i}' for i in range(1, 10)] + \
                           [f'id_{i}' for i in range(12, 39)]

    print("Handling missing values and encoding categoricals...")
    for col in train.columns:
        if col in ['TransactionID', 'isFraud', 'TransactionDT']:
            continue
            
        if col in categorical_features:
            train[col] = train[col].fillna('unknown').astype(str)
        else:
            train[col] = pd.to_numeric(train[col], errors='coerce')
            train[col] = train[col].fillna(0).astype('float32')

    # Encode categoricals
    for col in categorical_features:
        if col in train.columns:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col]).astype('int32')

    # Scaling numerical features
    print("Scaling numerical features...")
    num_cols = [c for c in train.columns if c not in categorical_features + ['TransactionID', 'isFraud', 'TransactionDT']]
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])

    train['isFraud'] = train['isFraud'].astype('int8')

    print("Saving processed train data...")
    # Save as parquet to be much faster and memory efficient when reading
    train.to_parquet('data/processed_train.parquet')
    
    print(f"Finished processing. Final shape: {train.shape}")
    print("Data saved to data/processed_train.parquet")

if __name__ == '__main__':
    preprocess_ieee_data()
