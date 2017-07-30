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
        return

def vectorize_dataset(df):
    X = np.zeros((len(df), len(VectorColumns)))
    Y = np.zeros(len(df))

