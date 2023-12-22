import pandas as pd

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


