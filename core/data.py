from enum import Enum

import pandas as pd
import numpy as np

class VectorColumns(Enum):
    HAS_AT_SIGN = 0
    PRE_AT_LENGTH = 1

def get_dataset_as_dataframe(datafile):
    return pd.read_csv(datafile, names=['email_address', 'is_spam'])

def has_at_sign(email_address):
    return ('@' in email_address)

def pre_at_length(email_address):
    if has_at_sign(email_address):
        return 1 if len(email_address[:email_address.index('@')]) > 2 else 0
    else:
        return 0

def vectorize_dataset(df):
    X = np.zeros((len(df), len(VectorColumns)))
    Y = np.zeros(len(df))

    for index, email_address, is_spam in df.itertuples():
        X[index][VectorColumns.HAS_AT_SIGN.value] = 1 if has_at_sign(email_address) else 0
        X[index][VectorColumns.PRE_AT_LENGTH.value] = pre_at_length(email_address)

        Y[index] = float(is_spam)

    return X, Y

def vectorize_from_string(email_address):
    X = np.zeros(len(VectorColumns))
    X[VectorColumns.HAS_AT_SIGN.value] = 1 if has_at_sign(email_address) else 0
    X[VectorColumns.PRE_AT_LENGTH.value] = pre_at_length(email_address)

    return X