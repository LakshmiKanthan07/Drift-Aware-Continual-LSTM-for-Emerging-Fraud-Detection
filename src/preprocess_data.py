import polars as pl
import pandas as pd
import json
import os
from sklearn.preprocessing import StandardScaler

def preprocess_ieee_data():
    print("Loading data with Polars...")
    # Read the data quickly using polars
    train_transaction = pl.read_csv('data/train_transaction.csv', infer_schema_length=10000, ignore_errors=True)
    train_identity = pl.read_csv('data/train_identity.csv', infer_schema_length=10000, ignore_errors=True)

    print(f"Transaction shape: {train_transaction.shape}")
    print(f"Identity shape: {train_identity.shape}")

    print("Merging data...")
    train = train_transaction.join(train_identity, on='TransactionID', how='left')
    del train_transaction
    del train_identity

    print(f"Merged train shape: {train.shape}")

    print("Sorting by time...")
    train = train.sort('TransactionDT')

    # Identify categorical columns
    categorical_features = ['ProductCD'] + \
                           [f'card{i}' for i in range(1, 7)] + \
                           ['addr1', 'addr2', 'P_emaildomain', 'R_emaildomain'] + \
                           [f'M{i}' for i in range(1, 10)] + \
                           [f'id_{i}' for i in range(12, 39)]

    print("Computing Velocity Features...")
    # Convert TransactionDT (seconds) to actual time duration for Polars rolling_sum
    train = train.with_columns(
        (pl.col("TransactionDT") * 1000000).cast(pl.Datetime).alias("dt_time")
    )

    # Compute Rolling features using DataFrame.rolling
    print("Computing 1H rolling features...")
    rolled_1h = train.rolling(index_column="dt_time", period="1h", group_by="card1").agg(
        pl.col("TransactionAmt").sum().alias("Amount_1H"),
        pl.len().alias("Count_1H")
    )
    
    print("Computing 24H rolling features...")
    rolled_24h = train.rolling(index_column="dt_time", period="24h", group_by="card1").agg(
        pl.col("TransactionAmt").sum().alias("Amount_24H"),
        pl.len().alias("Count_24H")
    )

    print("Joining features back...")
    # Join back (this broadcasts correctly even if multiple transactions exist at the exact same millisecond)
    train = train.join(rolled_1h, on=["card1", "dt_time"], how="left")
    train = train.join(rolled_24h, on=["card1", "dt_time"], how="left")


    print("Handling missing values and encoding categoricals...")
    cat_vocab = {}
    actual_cat_features = []
    
    # We will fall back to pandas for the sklearn parts to keep it identical to original
    df_pd = train.to_pandas()
    
    for col in df_pd.columns:
        if col in ['TransactionID', 'isFraud', 'TransactionDT', 'dt_time']:
            continue
            
        if col in categorical_features:
            df_pd[col] = df_pd[col].fillna('unknown').astype(str)
        else:
            df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce').fillna(0).astype('float32')

    from sklearn.preprocessing import LabelEncoder
    
    for col in categorical_features:
        if col in df_pd.columns:
            le = LabelEncoder()
            df_pd[col] = le.fit_transform(df_pd[col]).astype('int32')
            cat_vocab[col] = int(df_pd[col].max() + 1)
            actual_cat_features.append(col)

    print("Scaling numerical features...")
    # Add our new velocity features to the continuous columns
    new_num_cols = ["Amount_1H", "Amount_24H", "Count_1H", "Count_24H"]
    num_cols = [c for c in df_pd.columns if c not in categorical_features + ['TransactionID', 'isFraud', 'TransactionDT', 'dt_time']]
    
    scaler = StandardScaler()
    df_pd[num_cols] = scaler.fit_transform(df_pd[num_cols])

    df_pd['isFraud'] = df_pd['isFraud'].astype('int8')

    print("Saving processed train data and metadata...")
    df_pd.drop(columns=['dt_time'], inplace=True, errors='ignore')
    df_pd.to_parquet('data/processed_train.parquet')
    
    metadata = {
        'categorical_features': actual_cat_features,
        'numerical_features': num_cols,
        'cat_vocab': cat_vocab
    }
    with open('data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Finished processing. Final shape: {df_pd.shape}")
    print("Data saved to data/processed_train.parquet")
    print("Metadata saved to data/metadata.json")

if __name__ == '__main__':
    preprocess_ieee_data()
