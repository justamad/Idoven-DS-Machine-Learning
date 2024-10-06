import pandas as pd
import numpy as np
import wfdb
import ast

from os.path import join
from typing import List, Dict, Tuple


def load_physionet_dataset(path: str):
    Y = pd.read_csv(join(path, 'ptbxl_database.csv'), index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(join(path, 'scp_statements.csv'), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

    Y = Y[Y['diagnostic_superclass'].apply(lambda x: len(x) > 0)]
    return Y


def aggregate_diagnostic(y_dic: Dict, agg_df: pd.DataFrame) -> List:
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def prepare_training_data(path: str, Y: pd.DataFrame, sampling_rate=100) -> Tuple[np.ndarray, pd.DataFrame]:
    X = load_raw_ecg_data(Y, sampling_rate, path)
    return X, Y


def load_raw_ecg_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def prepare_ground_truth_data(df: pd.DataFrame):
    df["ground_truth"] = df['diagnostic_superclass'].apply(calculate_one_hot_encoding_per_sample)
    return df


def calculate_one_hot_encoding_per_sample(sample: List) -> List[int]:
    assert sample != [], f"Sample is empty: {sample}"
    if sample == ["NORM"]:
        return [0]
    return [1]


if __name__ == '__main__':
    df = load_physionet_dataset("../../data/physionet.org/files/ptb-xl/1.0.2/")
    print(df)
