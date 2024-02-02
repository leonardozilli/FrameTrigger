from datasets import load_dataset, Dataset
from copy import deepcopy
import pandas as pd
import numpy as np


def flatten(df):
    """combine a dataframe group to a one-line instance.

    Args:
        df ([dataframe]): a dataframe, represents a group

    Returns:
        a one-line dict
    """
    label_values = np.stack(df['frame_tags'].values)
    processed = np.zeros_like(label_values)

    for token_id in range(label_values.shape[1]):
        labels = label_values[:, token_id]
        for i in range(len(labels)):
            if labels[i] != 0:
                processed[i, token_id] = labels[i]
                break
    aggregated_tags = processed.sum(axis=0)
    result = df.iloc[0].to_dict()
    result['frame_tags'] = aggregated_tags

    return result


def combine(ds, group_column, ):
    ds = ds.to_pandas()
    combined = []
    for sent_id, group in ds.groupby(group_column):
        combined.append(flatten(group))
    return Dataset.from_pandas(pd.DataFrame(combined))


def load_dataset_hf(flatten=True):
    ds = load_dataset('liyucheng/FrameNet_v17')
    if flatten:
        flat_ds = deepcopy(ds)
        for k, v in flat_ds.items():
            flat_ds[k] = combine(v, 'sent_id')
        return flat_ds
    else:
        return ds


