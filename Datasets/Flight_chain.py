import warnings
import datetime
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from tqdm import tqdm
import warnings
import datetime
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset
from tqdm import tqdm
import os

# Configuration
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50


def process_year_data(year):
    """Process data for a single year"""
    print(f"Start processing data for {year}...")
    start_time = time.time()

    # 1. Data loading and preprocessing
    file_path = f"flight_with_weather_{year}.csv"
    df = pd.read_csv(file_path, low_memory=False)

    # Datetime processing
    datetime_cols = ['FL_DATE', 'CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME']
    df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime, errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df.rename(columns={'FL_DATE': 'DATE'}, inplace=True)

    # Feature engineering
    df['CRS_DEP_TIME_HOUR'] = df['CRS_DEP_TIME'].dt.hour.astype('int8')
    df['CRS_ARR_TIME_HOUR'] = df['CRS_ARR_TIME'].dt.hour.astype('int8')
    df = df.sort_values(by='DATE').reset_index(drop=True)

    # Encode categorical features
    encoder = LabelEncoder()
    df['OP_CARRIER'] = encoder.fit_transform(df['OP_CARRIER'])
    df['OP_CARRIER_FL_NUM'] = encoder.fit_transform(df['OP_CARRIER_FL_NUM'])

    # 2. Dataset splitting
    date_dim = df[['DATE']].drop_duplicates()
    date_dim['DAY_OF_YEAR'] = date_dim['DATE'].dt.dayofyear
    date_dim['MONTH'] = date_dim['DATE'].dt.month

    rng = np.random.RandomState(seed=42)
    train_days, valid_days, test_days = [], [], []

    for month in sorted(date_dim['MONTH'].unique()):
        month_dates = date_dim[date_dim['MONTH'] == month]['DAY_OF_YEAR']
        n_days = len(month_dates)

        if n_days < 3:
            alloc_days = list(month_dates) * 3
            mandatory_days = alloc_days[:3]
        else:
            mandatory_days = rng.choice(month_dates, 3, replace=False)

        train_days.append(mandatory_days[0])
        valid_days.append(mandatory_days[1])
        test_days.append(mandatory_days[2])

        remaining_days = [d for d in month_dates if d not in mandatory_days]
        n_remaining = len(remaining_days)

        if n_remaining > 0:
            permuted = rng.permutation(remaining_days)
            split1 = int(round(n_remaining * 0.6))
            split2 = split1 + int(round(n_remaining * 0.2))

            train_days.extend(permuted[:split1])
            valid_days.extend(permuted[split1:split2])
            test_days.extend(permuted[split2:])

    date_dim['SET_TYPE'] = np.select(
        [date_dim['DAY_OF_YEAR'].isin(train_days),
         date_dim['DAY_OF_YEAR'].isin(valid_days),
         date_dim['DAY_OF_YEAR'].isin(test_days)],
        ['train', 'valid', 'test'],
        default='undefined'
    )

    df = pd.merge(df, date_dim[['DATE', 'SET_TYPE']], on='DATE', how='left')

    # 3. Create flight chains
    def create_flight_chains(data):
        data_sorted = data.sort_values(by=['OP_CARRIER', 'OP_CARRIER_FL_NUM', 'DATE', 'CRS_DEP_TIME'])
        grouped = data_sorted.groupby(['OP_CARRIER', 'OP_CARRIER_FL_NUM', 'DATE'])
        return {name: group for name, group in grouped}

    train_chains = create_flight_chains(df[df['SET_TYPE'] == 'train'])
    valid_chains = create_flight_chains(df[df['SET_TYPE'] == 'valid'])
    test_chains = create_flight_chains(df[df['SET_TYPE'] == 'test'])

    # 4. Process flight chain data
    def process_chain(chain, max_sequence_length=6):
        # Dense features
        dense_feat_names = ['O_TEMP', 'D_TEMP', 'O_PRCP', 'D_PRCP', 'O_WSPD', 'D_WSPD', 'FLIGHTS']
        dense_tensors = [
            torch.tensor(chain[name].fillna(0).values, dtype=torch.float32)
            for name in dense_feat_names
        ]
        dense_feat = torch.stack(dense_tensors, dim=1)

        # Sparse features
        sparse_feat_names = ['MONTH', 'DAY_OF_WEEK', 'CRS_ARR_TIME_HOUR',
                             'CRS_DEP_TIME_HOUR', 'ORIGIN_INDEX', 'DEST_INDEX',
                             'OP_CARRIER', 'OP_CARRIER_FL_NUM']

        chain['MONTH'] = (chain['MONTH'] - 1).astype(np.int16)
        chain['DAY_OF_WEEK'] = (chain['DAY_OF_WEEK'] - 1).astype(np.int16)

        sparse_tensors = [
            torch.tensor(chain[name].values.astype(np.int16), dtype=torch.int16)
            for name in sparse_feat_names
        ]
        sparse_feat = torch.stack(sparse_tensors, dim=1)

        # Labels and delays
        delays = torch.tensor(
            chain[['ARR_DELAY', 'DEP_DELAY']].values.astype(np.int16),
            dtype=torch.int16
        )

        labels = torch.tensor(
            np.column_stack((
                (chain['ARR_DELAY'] > 15).astype(np.int8),
                (chain['DEP_DELAY'] > 15).astype(np.int8)
            )),
            dtype=torch.int8
        )

        # Sequence length processing
        valid_len = min(len(dense_feat), max_sequence_length)

        # Unified truncation/padding
        def adjust_sequence(data, max_len):
            if len(data) < max_len:
                pad_shape = (max_len - len(data), data.shape[1])
                return torch.cat([data, torch.zeros(pad_shape, dtype=data.dtype)], dim=0)
            return data[:max_len]

        dense_feat = adjust_sequence(dense_feat, max_sequence_length)
        sparse_feat = adjust_sequence(sparse_feat, max_sequence_length)
        labels = adjust_sequence(labels, max_sequence_length)
        delays = adjust_sequence(delays, max_sequence_length)

        return dense_feat, sparse_feat, labels, valid_len, delays

    def process_all_chains(flight_chains):
        processed = []
        for name, chain in tqdm(flight_chains.items(), desc=f'Processing {year} chains'):
            processed.append(process_chain(chain))
        return processed

    # Process all datasets
    train_processed = process_all_chains(train_chains)
    valid_processed = process_all_chains(valid_chains)
    test_processed = process_all_chains(test_chains)

    # Create datasets
    def create_dataset(processed_data):
        dense = torch.stack([item[0] for item in processed_data])
        sparse = torch.stack([item[1] for item in processed_data])
        labels = torch.stack([item[2] for item in processed_data])
        valid_lens = torch.tensor([item[3] for item in processed_data], dtype=torch.long)
        delays = torch.stack([item[4] for item in processed_data])
        return TensorDataset(dense, sparse, labels, valid_lens, delays)

    train_dataset = create_dataset(train_processed)
    valid_dataset = create_dataset(valid_processed)
    test_dataset = create_dataset(test_processed)

    # 5. Save results
    output_dir = f"processed_data_{year}"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(train_dataset, os.path.join(output_dir, f'train_flight_chain_{year}.pt'))
    torch.save(valid_dataset, os.path.join(output_dir, f'val_flight_chain_{year}.pt'))
    torch.save(test_dataset, os.path.join(output_dir, f'test_flight_chain_{year}.pt'))

    elapsed = time.time() - start_time
    print(f"Finished processing data for {year}, time elapsed: {elapsed:.2f} seconds")
    return year


if __name__ == "__main__":
    import time

    years = range(2016, 2025)  # 2016 to 2024
    start_time = time.time()

    # Use multiprocessing for parallel processing
    with ProcessPoolExecutor(max_workers=min(4, len(years))) as executor:
        results = list(executor.map(process_year_data, years))

    total_time = time.time() - start_time
    print(f"All years processed! Total time: {total_time / 60:.2f} minutes")
    print("Processed years:", results)

