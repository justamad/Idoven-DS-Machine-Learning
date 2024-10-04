import pandas as pd
import numpy as np
import wfdb
import ast

from os.path import join
from typing import List, Dict, Tuple

POSSIBLE_GROUND_TRUTH_LABELS = ["NORM", "MI", "STTC", "HYP", "CD"]


def load_physionet_dataset(path: str, sampling_rate: int = 100, max_idx: int = 100):
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
    Y["ground_truth"] = Y.diagnostic_superclass.apply(calculate_one_hot_encoding_per_sample)
    # Y = Y.iloc[:max_idx]
    X = load_raw_ecg_data(Y, sampling_rate, path)
    return X, Y


def load_raw_ecg_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.ndarray:
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


def calculate_one_hot_encoding_per_sample(sample: List) -> np.ndarray:
    assert sample != [], f"Sample is empty: {sample}"
    one_hot = np.zeros(len(POSSIBLE_GROUND_TRUTH_LABELS), dtype=int)
    for value in sample:
        if value in POSSIBLE_GROUND_TRUTH_LABELS:
            one_hot[POSSIBLE_GROUND_TRUTH_LABELS.index(value)] = 1
    return one_hot
