import numpy as np
import pandas as pd
from experiment_1.runs.experiments import generate_dataset_json
# dataset
train = dict()

train["category"] = {str(k): k for k in range(10)}
train["color"] = {str(k): None for k in range(10)}
train["brightness"] = {str(k): "light" for k in range(10)}
train["size"] = {"0": "small",
                 "1": "small",
                 "2": "small",
                 "3": "small",
                 "4": "medium",
                 "5": "medium",
                 "6": "medium",
                 "7": "large",
                 "8": "large",
                 "9": "large"}
train["contour"] = {str(k): False for k in range(10)}

test = dict()
test["category"] = {str(k): k for k in range(10)}
test["color"] = {str(k): None for k in range(10)}
test["brightness"] = {str(k): "light" for k in range(10)}
test["size"] = {"0": "medium",
                "1": "medium",
                "2": "medium",
                "3": "medium",
                "4": "large",
                "5": "large",
                "6": "large",
                "7": "small",
                "8": "small",
                "9": "small"}
test["contour"] = {str(k): False for k in range(10)}


def build_dataframe(dct_dataset):
    """
    Here we build a generic dataframe given the object attributes.
    """
    df = pd.DataFrame(np.zeros((len(dct_dataset["category"]),
                                len(dct_dataset.keys())),
                               dtype=int),
                      columns=dct_dataset.keys())
    for k_ in dct_dataset.keys():
        for i_ in dct_dataset[k_]:
            df[k_][i_] = (dct_dataset[k_][i_])

    df = pd.DataFrame([dct_dataset[k_].values() for k_ in dct_dataset.keys()],
                      index=dct_dataset.keys()).T

    return df


def build_tr_vl():
    return build_dataframe(train), build_dataframe(test)
# we must check if these exist already or not

train_df, test_df = build_tr_vl()

generate_dataset_json(train_df, test_df, output_path="./../data_generation/")
