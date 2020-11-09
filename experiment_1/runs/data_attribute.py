import os
import numpy as np
import pandas as pd
import json
from os.path import join
from runs.experiments import generate_dataset_json
from runs.mnist_attribute_generation import transform
# dataset

train = dict()

# TODO: GENERATE VOCABULARY
# color = red, green, blue, pink, yellow
# questions -- category(0-9), color(10-14), brightness(15-17), size(18-20), contour(21-22)

vocab = {}
train["category"] = {str(k): k for k in range(10)}
train["color"] = {str(k): None for k in range(10)}
train["brightness"] = {str(k): "light" for k in range(10)}
train["size"] = {"0": "small",
                 "1": "small",
                 "2": "small",
                 "3": "small",
                 "4": "small",
                 "5": "small",
                 "6": "small",
                 "7": "small",
                 "8": "small",
                 "9": "small"}
train["contour"] = {str(k): False for k in range(10)}

test = dict()
test["category"] = {str(k): k for k in range(10)}
test["color"] = {str(k): None for k in range(10)}
test["brightness"] = {str(k): "light" for k in range(10)}
test["size"] = {"0": "large",
                "1": "large",
                "2": "large",
                "3": "large",
                "4": "large",
                "5": "large",
                "6": "large",
                "7": "large",
                "8": "large",
                "9": "large"}
test["contour"] = {str(k): False for k in range(10)}

# questions -- category(0-9), color(10-14), brightness(15-17), size(18-20), contour(21)
vocab["question_token_to_idx"] = [k_ for k_ in range(10)]  # +  # add other lists
vocab["answer_idx_to_token"] = {"true": 1,
                                "false": 0}


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


def generate_data_matrix(df_object_specs,
                         split_folder,
                         flag_program=False):
    """
    This function saves a tuple of the input we need to train and test the
    networks.
    :param df_object_specs: characteristics about the dataset
    :param split_folder: where to find the original split for MNIST
    :param flag_program: if we want to save the programs or not
    :returns: None
    original:
        (questions, images, feats, answers, programs_exec, programs_json)
    if flag_program is False
    for training and for test
        (questions, feats, answers, programs_exec, programs_json)
    else
        for training and for test
        (questions, feats, answers)
    It saves in files the two.
    """
    path_output_folder = df_object_specs["dataset_path"]

    x_split_files = ["x_train.npy", "x_valid.npy", "x_test.npy"]

    # we load the train, validation and test mnist (x,y)
    x_original = [np.load(join(split_folder, file_)) for file_ in x_split_files]
    y_original = [np.load(join(split_folder, "y_%s" % file_.split("_")[-1])) for file_ in x_split_files]
    n_new = [x_.shape[0] for x_ in x_original]  # number of points per split
    _, dim_x, dim_y = x_original[0].shape  # image dimensions (equivalent across splits)
    # we check if we have no colors, to have the proper tensor shape
    flag_no_color = (all(v is None for v in df_object_specs['dataset_train']['color'].values()) and
                     all(v is None for v in df_object_specs['dataset_test']['color'].values()))

    x_new = [np.zeros((n_, dim_x, dim_y)) for n_ in n_new] if flag_no_color else [np.zeros((n_, 3, dim_x, dim_y))
                                                                                  for n_ in n_new]
    questions_new = [np.zeros(n_, dtype=int) for n_ in n_new]
    answers_new = [np.zeros(n_, dtype=int) for n_ in n_new]
    split_configuration = ["dataset_%s" % split for split in ["train", "test", "test"]]

    # train, validation, and test
    for id_, (x_, y_, n_, split_config_, split_file_) in enumerate(zip(x_original,
                                                                       y_original,
                                                                       n_new,
                                                                       split_configuration,
                                                                       x_split_files)):
        for i_ in range(np.unique(y_).size):  # for every category
            if vocab["question_token_to_idx"] == [k_ for k_ in range(10)]:  # only object category
                bm = y_ == i_  # bool mask for a specific category
                pos_id = np.argwhere(bm).reshape(-1,)
                np.random.shuffle(pos_id)
                correct_answers_id = pos_id[:pos_id.size//2]
                incorrect_answers_id = pos_id[pos_id.size//2:]
                # we sample all the possible questions but from the right category
                tmp_questions = np.random.choice(np.delete(np.arange(np.unique(y_).size),
                                                           i_), size=incorrect_answers_id.size)
                for j_ in pos_id:  # and we assign
                    x_new[id_][j_] = transform(x_[j_] / 255,
                                               reshape=df_object_specs[split_config_]['size'][i_],
                                               color=df_object_specs[split_config_]['color'][i_],
                                               bright=df_object_specs[split_config_]['brightness'][i_],
                                               contour=df_object_specs[split_config_]['contour'][i_])
                questions_new[id_][correct_answers_id] = i_
                questions_new[id_][incorrect_answers_id] = tmp_questions
                answers_new[id_][correct_answers_id] = 1
                answers_new[id_][incorrect_answers_id] = 0
            else:
                raise ValueError("Not implemented yet")
                # check the values for the vocabulary


        np.save(join(path_output_folder, "feats_%s" % split_file_.split("_")[-1]), x_new[id_])
        np.save(join(path_output_folder, "questions_%s" % split_file_.split("_")[-1]), questions_new[id_])
        np.save(join(path_output_folder, "answers_%s" % split_file_.split("_")[-1]), answers_new[id_])

    # save vocab
    with open(join(path_output_folder, 'vocab.json'), 'w') as outfile:
        json.dump(vocab, outfile)


def build_tr_vl():
    return build_dataframe(train), build_dataframe(test)
    # we must check if these exist already or not


def generate_data_file(output_data_folder,
                       splits_folder):
    train_df, test_df = build_tr_vl()
    out_dict = generate_dataset_json(train_df,
                                     test_df,
                                     output_data=output_data_folder)
    # we need this to be the new df
    if out_dict is not None:
        os.makedirs(out_dict["dataset_path"], exist_ok=True)
        generate_data_matrix(out_dict,
                             splits_folder,
                             flag_program=False)
    return
