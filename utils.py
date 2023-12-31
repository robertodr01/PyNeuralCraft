import pandas as pd
import mlp as m

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


def k_fold_cross_validation(model: m.MLP, X: pd.DataFrame, y: pd.DataFrame, k: int):

    subset_dim = len(X) // k
    
    cv_scores = []

    for i in range(k):
        start = i * subset_dim
        end = (i + 1) * subset_dim

        X_train = pd.concat([X.iloc[:start,:], X.iloc[end:,:]])
        y_train = pd.concat([y.iloc[:start,:], y.iloc[end:,:]])
        X_val = X.iloc[start:end]
        y_val = y.iloc[start:end]

        # model.fit(X_train, y_train)

        # score validation
        # cv_scores.append(model.score(X_val, y_val))
        
    # return cv_scores
