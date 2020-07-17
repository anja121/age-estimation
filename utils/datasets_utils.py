import random
import pandas as pd
import numpy as np


def load_csv(csv_path, shuffle=True, seed=None):
    """
    :param csv_path: string; path to csv annotation file
    :param shuffle: bool;
    :param seed: int;
    :return: pandas DataFrame
    """
    data_df = pd.read_csv(csv_path, index_col=0)

    if shuffle:
        seed = seed if seed is not None else random.randint(0, 1000)
        np.random.seed(seed)
        data_df = data_df.reindex(np.random.permutation(data_df.index))

    return data_df


def train_valid_split(data_df, train_percent, seed=None):
    """
    Splitting dataset to train and valid with respect to train_percent value
    :param data_df: pandas Dataframe;
    :param train_percent: float;
    :param seed: int;
    :return: pandas DataFrame train_data, pandas Dataframe valid_data;
    """
    seed = seed if seed is not None else random.randint(0, 1000)

    valid_data = data_df.sample(n=int((1-train_percent)*len(data_df)), random_state=seed)
    train_data = data_df.drop(valid_data.index)

    return train_data, valid_data
