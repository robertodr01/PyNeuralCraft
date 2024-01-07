import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_splits(df: pd.DataFrame, train_percentage: float, test_percentage: float, validation_percentage: float = 0):

    if train_percentage + test_percentage + validation_percentage != 1:
        raise ValueError('Error on total percentage, it`s not 1!')
        
    train_percentage = train_percentage * 100
    test_percentage = test_percentage * 100
    validation_percentage = validation_percentage * 100
    
    n_instances = len(df)

    train_split = int(n_instances * train_percentage / 100)
    remaining_split = n_instances - train_split
    test_split = remaining_split - int(n_instances * validation_percentage / 100)
    val_split = remaining_split - test_split

    train_df = df[: train_split]
    test_df = df[train_split: train_split + test_split]
    validation_df = df[train_split + test_split : ]

    return train_df, test_df, validation_df

def hold_out_validation(X: np.ndarray, y: np.ndarray):
    train_len = round((len(X) / 100) * 80)
    dataset = []
    obj = {}

    obj['X_train'] = X[:train_len]
    obj['y_train'] = y[:train_len]
    obj['X_val']  = X[train_len:]
    obj['y_val']  = y[train_len:]
    dataset.append(obj)
    return dataset

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k: int):

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    subset_dim = len(X) // k
    dataset = []

    for i in range(k):
        obj = {}

        start = i * subset_dim
        end = (i + 1) * subset_dim

        X_train = pd.concat([X.iloc[:start,:], X.iloc[end:,:]])
        y_train = pd.concat([y.iloc[:start,:], y.iloc[end:,:]])
        X_val = X.iloc[start:end]
        y_val = y.iloc[start:end]

        obj["X_train"] = X_train.to_numpy()
        obj["y_train"] = y_train.to_numpy()
        obj["X_val"] = X_val.to_numpy()
        obj["y_val"] = y_val.to_numpy()

        dataset.append(obj)
    return dataset