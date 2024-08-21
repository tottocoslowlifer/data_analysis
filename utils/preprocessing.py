import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset


def to_torch(X: np.ndarray, y: np.ndarray):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y


def prepare_nn(main_df, test_size_1, test_size_2):
    std_x = StandardScaler()
    std_y = StandardScaler()
    X = std_x.fit_transform(main_df.drop("observed", axis=1))
    y = std_y.fit_transform(main_df[["observed"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_1, shuffle=False
    )
    X_valid, X_eval, y_valid, y_eval = train_test_split(
        X_test, y_test, test_size=test_size_2, shuffle=False
    )
    train_df = pd.DataFrame(
        np.hstack((y_train, X_train)), columns=main_df.columns
    )
    valid_df = pd.DataFrame(
        np.hstack((y_valid, X_valid)), columns=main_df.columns
    )
    eval_df = pd.DataFrame(
        np.hstack((y_eval, X_eval)), columns=main_df.columns
    )

    # Tensor型に変換
    X_train, y_train = to_torch(X_train, y_train)
    X_valid, y_valid = to_torch(X_valid, y_valid)
    X_eval, y_eval = to_torch(X_eval, y_eval)

    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    out = {
        "scaler": std_y,
        "train_data": train_df,
        "train_dataset": train_dataset,
        "valid_data": valid_df,
        "valid_dataset": valid_dataset,
        "eval_data": eval_df,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_eval": X_eval,
        "y_eval": y_eval
    }

    return out
