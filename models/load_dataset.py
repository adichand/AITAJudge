import pandas as pd
from sklearn.model_selection import train_test_split


# Load the dataset
def get_dataframe(DATASET, TEXT_COL, LABEL_COL, dataset_size=1000, ignore_removed=True):
    data = pd.read_csv(DATASET, nrows=dataset_size)

    if ignore_removed:
        data = data[~data[TEXT_COL].str.contains('removed')]

    return data


def get_train_test_split(data, TEXT_COL, LABEL_COL):
    X_train, X_test, y_train, y_test = train_test_split(data[TEXT_COL], data[LABEL_COL], test_size=0.2)
    return X_train, X_test, y_train, y_test