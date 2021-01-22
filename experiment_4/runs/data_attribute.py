import os
import numpy as np
import pandas as pd
import json
from os.path import join
from runs.experiments import generate_dataset_json
from runs.mnist_attribute_generation import transform


# TODO: GENERATE VOCABULARY
# color = red, green, blue, pink, yellow
# questions -- category(0-9), color(10-14), brightness(15-17), size(18-20), contour(21-22)

vocab = {}

TRAIN = dict()
TRAIN["category"] = {str(k): k for k in range(10)}
TRAIN["color"] = {str(k): None for k in range(10)}
TRAIN["brightness"] = {str(k): "light" for k in range(10)}
TRAIN["size"] = {"0": "small",
                 "1": "small",
                 "2": "small",
                 "3": "small",
                 "4": "small",
                 "5": "small",
                 "6": "small",
                 "7": "small",
                 "8": "small",
                 "9": "small"}
TRAIN["contour"] = {str(k): False for k in range(10)}

TEST = dict()
TEST["category"] = {str(k): k for k in range(10)}
TEST["color"] = {str(k): None for k in range(10)}
TEST["brightness"] = {str(k): "light" for k in range(10)}
TEST["size"] = {"0": "large",
                "1": "large",
                "2": "large",
                "3": "large",
                "4": "large",
                "5": "large",
                "6": "large",
                "7": "large",
                "8": "large",
                "9": "large"}
TEST["contour"] = {str(k): False for k in range(10)}

# questions -- category(0-9), color(10-14), brightness(15-17), size(18-20), contour(21)
# category          0-9
# color             10: R, 11: G, 12: B, 13: yellow, 14: purple
# brightness        15: dark, 16: half, 17: light
# size              18: small, 19: medium, 20: large
# contour           21: bool (if False, no full digit)
category_idx = [k_ for k_ in range(10)]
color_idx = [k_ for k_ in range(10, 15)]
brightness_idx = [k_ for k_ in range(15, 18)]
size_idx = [k_ for k_ in range(18, 21)]
contour_idx = [21]

vocab["all_questions_token_to_idx"] = {"%i" % k_: k_ for k_ in category_idx}  # +  # add other lists
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_ for (s_, k_) in zip(["red", "green", "blue", "yellow", "purple"],
                                                                              color_idx)})
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_ for (s_, k_) in zip(["dark", "half", "light"],
                                                                              brightness_idx)})
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_ for (s_, k_) in zip(["small", "medium", "large"],
                                                                              size_idx)})
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_ for (s_, k_) in zip(["contour"], contour_idx)})

vocab["answer_token_to_idx"] = {"true": 1,
                                "false": 0}

vocab["question_token_to_idx"] = {"%i" % k_: k_ for k_ in category_idx}
vocab["question_token_to_idx"].update({"%s" % s_: k_ for (s_, k_) in zip(["small", "medium", "large"],
                                                                          size_idx)})


def find_categories_given_attributes(dct, attribute_family, attribute):
    list_elements = []
    for k_, v_, in dct[attribute_family].items():
        if type(v_) is list:
            if attribute in v_:
                list_elements.append(k_)
        elif v_ == attribute:
            list_elements.append(k_)
    return list_elements


def map_question_idx_to_attribute_category(idx):
    if idx in category_idx:
        return 0
    elif idx in color_idx:
        return 1
    elif idx in brightness_idx:
        return 2
    elif idx in size_idx:
        return 3
    elif idx in contour_idx:
        return 4
    else:
        raise ValueError("The inserted index is wrong")


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


class ObjectImage(object):
    def __init__(self,
                 category=None,
                 color=None,
                 brightness=None,
                 size=None,
                 contour=None,
                 attribute=None,
                 vocab=vocab,
                 split="train"):
        """If we fix one of its attribute,
        we must look for those that are not fixed"""

        self.attribute_family = attribute
        self.vocab = vocab

        flag_no_category = True
        if self.attribute_family == "color":
            self.color = color
            attribute_name = "color"
            attribute_instance = color
        elif self.attribute_family == "brightness":
            self.brightness = brightness
            attribute_name = "brightness"
            attribute_instance = brightness
        elif self.attribute_family == "size":
            self.size = size
            attribute_name = "size"
            attribute_instance = size
        elif self.attribute_family == "contour":
            self.contour = contour
            attribute_name = "contour"
            attribute_instance = contour
        else:
            flag_no_category = False

        if flag_no_category:
            if self.category is None:
                if split == "train":
                    list_categories = find_categories_given_attributes(train,
                                                                       attribute_name,
                                                                       attribute_instance)
                else:
                    list_categories = find_categories_given_attributes(test,
                                                                       attribute_name,
                                                                       attribute_instance)
        else:
            self.category = self.category
            for (o_, s_) in zip([color, brightness, size, contour],
                                ["color", "brightness", "size", "contour"]):
                if o_ is None:
                    if split == "train":
                        self.o_ = [[train][s_]["%i" %k_] for k_ in self.category]
                    else:
                        self.o_ = [test[s_]["%i" % k_] for k_ in self.category]


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
        """for i_ in range(np.unique(y_).size):  # for every category
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
        else:"""
        answer_vec = np.array([], dtype=int)
        question_vec = np.array([], dtype=int)
        n_questions = len(vocab["question_token_to_idx"].values())
        object_specs = np.array([], dtype=int)  # category, color, bright, size, contour

        for id_q, (q_k, q_v) in enumerate(vocab["question_token_to_idx"].items()):  # int index
            examples_per_question = n_ / n_questions  # per each question

            tmp = np.zeros(examples_per_question, dtype=int)
            tmp[1::2] = 1
            answer_vec = np.append(answer_vec, tmp)  # balanced

            question_vec = np.append(question_vec, q_v * np.ones(examples_per_question, dtype=int))
            tmp = np.zeros((examples_per_question, 5), dtype=int)
            index = map_question_idx_to_attribute_category(q_v)
            tmp[1::2, index] = q_v
            object_specs = np.append(object_specs, tmp)

            # we need to adjust the others
            if index == 0:
                # randomly sampling from 0 - 9 avoiding the question
                tmp[::2, index] = np.array([np.random.choice(np.delete(np.arange(10), 1), size=9, replace=False)
                                            for kkk_ in range(examples_per_question // 2)]).reshape(-1,)
                for kk_, cat_ in enumerate(tmp[:, index]):
                    cat2 =
                    tmp[kk_, 1]
                    tmp[kk_, 1]
                    tmp[kk_, 1]

            elif index == 1:  # color
            elif index == 2:  # brightness
            elif index == 3:  # size
            else:  # contour


            for jjj in range(examples_per_question):
                if q_ in category_idx and jjj % 2:  # yes answer
                    object_specs["%s" % id_q]["category"] = q_
                elif q_ in category_idx and not jjj % 2:  # yes answer
                    new_category = np.delete(category_idx, q_)[jjj % (len(category_idx)-1)]
                    object_specs["%s" % id_q]["category"] = new_category
                elif q_ in color_idx and jjj % 2:
                    object_specs["%s" % id_q]["color"] = q_


                category_idx = [k_ for k_ in range(10)]
                color_idx = [k_ for k_ in range(10, 15)]
                brightness_idx = [k_ for k_ in range(15, 18)]
                size_idx = [k_ for k_ in range(18, 21)]
                contour_idx = [21]
            answer =examples_per_question //2

            output_sizes = df_object_specs[split_config_]['size'][i_][(id_j_ + np.random.randint(1234))
                                                                      % n_sizes]



        # this is valid for all the three splits
        divide_dataset = 0
        flag_category = all(item in vocab["question_token_to_idx"]
                            for item in category_idx)
        if flag_category:
            divide_dataset += len(category_idx)

        flag_color = all(item in vocab["question_token_to_idx"]
                         for item in color_idx)
        if flag_color:
            divide_dataset += len(color_idx)

        flag_brightness = all(item in vocab["question_token_to_idx"]
                              for item in brightness_idx)
        if flag_brightness:
            divide_dataset += len(brightness_idx)

        flag_size = all(item in vocab["question_token_to_idx"]
                        for item in size_idx)
        if flag_size:
            divide_dataset += len(size_idx)

        flag_contour = all(item in vocab["question_token_to_idx"]
                           for item in contour_idx)
        if flag_contour:
            divide_dataset += 1  # there's only one question in this case
            # raise ValueError("Not implemented yet")
            # check the values for the vocabulary

        # divide dataset tells us how many possible questions we have
        # we need n_data_points // divide_dataset examples per question
        # if flag_category, then we group elements by their class category
        # and the first n_data_points // divide_dataset are questions

        if flag_category:
            # TODO: define attributes as a list instead of single strings
            # TODO: question: how do we make it balanced?
            # we need to save the attributes for all, and then ask based on those
            for i_ in range(np.unique(y_).size):
                bm = y_ == i_  # bool mask for a specific category
                pos_id = np.argwhere(bm).reshape(-1, )
                np.random.shuffle(pos_id)
                fraction = np.sum(bm) // (divide_dataset+1-10)  # 10 are the object categories
                correct_answers_id = pos_id[: fraction//2]
                incorrect_answers_id = pos_id[fraction//2:fraction]

                for id_j_, j_ in enumerate(pos_id):  # and we assign
                    if flag_color:  # if the object is colored
                        n_colors = len(color_idx) if flag_color else 1
                        # we give as a color one of the element in the list of colors
                        output_color = df_object_specs[split_config_]['color'][i_][(id_j_ + np.random.randint(1234))
                                                                                   % n_colors]
                    else:
                        # otherwise, we are going to give the color that have been assigned for that category
                        output_color = df_object_specs[split_config_]['color'][i_]

                    if flag_size:
                        n_sizes = len(size_idx)
                        output_sizes = df_object_specs[split_config_]['size'][i_][(id_j_ + np.random.randint(1234))
                                                                                  % n_sizes]
                    else:
                        output_sizes = df_object_specs[split_config_]['size'][i_]

                    if flag_brightness:
                        n_brightness = len(brightness_idx)
                        output_brightness = df_object_specs[split_config_]['brightness'][i_][(id_j_ + np.random.randint(1234))
                                                                                             % n_brightness]
                    else:
                        output_brightness = df_object_specs[split_config_]['brightness'][i_]

                    if flag_contour:
                        n_contour = len(contour_idx)  # is it wo or w contour
                        output_contour = df_object_specs[split_config_]['contour'][i_][(id_j_ + np.random.randint(1234))
                                                                                       % n_contour]
                    else:
                        output_contour = df_object_specs[split_config_]['contour'][i_]

                    x_new[id_][j_] = transform(x_[j_] / 255,
                                               reshape=output_sizes,
                                               color=output_color,
                                               bright=output_brightness,
                                               contour=output_contour)

                # we sample all the possible questions but from the right category
                tmp_questions = np.random.choice(np.delete(np.arange(np.unique(y_).size),
                                                           i_), size=incorrect_answers_id.size)
                questions_new[id_][correct_answers_id] = i_
                questions_new[id_][incorrect_answers_id] = tmp_questions
                answers_new[id_][correct_answers_id] = 1
                answers_new[id_][incorrect_answers_id] = 0

        np.save(join(path_output_folder, "feats_%s" % split_file_.split("_")[-1]), x_new[id_])
        np.save(join(path_output_folder, "questions_%s" % split_file_.split("_")[-1]), questions_new[id_])
        np.save(join(path_output_folder, "answers_%s" % split_file_.split("_")[-1]), answers_new[id_])

    # save vocab
    with open(join(path_output_folder, 'vocab.json'), 'w') as outfile:
        json.dump(vocab, outfile)


def build_tr_vl():
    return build_dataframe(TRAIN), build_dataframe(TEST)
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
