import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
from os.path import join
from functools import reduce
import h5py
from runs.experiments import generate_dataset_json
from runs.mnist_attribute_generation import transform


# TODO: GENERATE VOCABULARY
# color = red, green, blue, pink, yellow
# questions -- category(0-9), color(10-14), brightness(15-17), size(18-20), contour(21-22)

vocab = {}
TRAIN = dict()
TRAIN["category"] = {str(k): k for k in range(10)}
TRAIN["color"] = {"0": ["red", "green"],
                  "1": ["red", "green"],
                  "2": ["green", "yellow"],
                  "3": ["green", "yellow"],
                  "4": ["yellow", "blue"],
                  "5": ["yellow", "blue"],
                  "6": ["blue", "purple"],
                  "7": ["blue", "purple"],
                  "8": ["purple", "red"],
                  "9": ["purple", "red"]}
TRAIN["brightness"] = {str(k): "light" for k in range(10)}
TRAIN["size"] = {"0": ["small", "large"],
                 "1": "large",
                 "2": "small",
                 "3": ["large", "small"],
                 "4": "small",
                 "5": "large",
                 "6": "small",
                 "7": "small",
                 "8": "large",
                 "9": 'large'}
TRAIN["contour"] = {str(k): [False, True] for k in range(10)}

TEST = dict()
TEST["category"] = {str(k): k for k in range(10)}
TEST["color"] = {"0": ["green", "yellow"],
                 "1": ["green", "yellow"],
                 "2": ["red", "green"],
                 "3": ["red", "green"],
                 "4": ["purple", "red"],
                 "5": ["purple", "red"],
                 "6": ["yellow", "blue"],
                 "7": ["yellow", "blue"],
                 "8": ["blue", "purple"],
                 "9": ["blue", "purple"]}
# {str(k): None for k in range(10)}
TEST["brightness"] = {str(k): "light" for k in range(10)}
TEST["size"] = {"0": "small",
                "1": "large",
                "2": "small",
                "3": "large",
                "4": "small",
                "5": "large",
                "6": "small",
                "7": "large",
                "8": "small",
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

question_families = {"category": category_idx,
                     "color": color_idx,
                     "brightness": brightness_idx,
                     "size": size_idx,
                     "contour": contour_idx}

vocab["all_questions_token_to_idx"] = {"%i" % k_: k_ for k_ in category_idx}  # +  # add other lists
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_
                                            for (s_, k_) in zip(["red",
                                                                 "green",
                                                                 "blue",
                                                                 "yellow",
                                                                 "purple"],
                                                                 color_idx)})
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_
                                            for (s_, k_) in zip(["dark",
                                                                 "half",
                                                                 "light"],
                                                                brightness_idx)})
vocab["all_questions_token_to_idx"].update({"%s" % s_: k_
                                            for (s_, k_) in zip(["small",
                                                                 "medium",
                                                                 "large"],
                                                                size_idx)})
vocab["question_token_to_idx"] = vocab["all_questions_token_to_idx"].copy()

vocab["all_questions_token_to_idx"].update({"%s" % s_: k_
                                            for (s_, k_) in zip(["contour"], contour_idx)})

vocab["answer_token_to_idx"] = {"true": 1,
                                "false": 0}


def invert_dict(d):
    return {v: k for k, v in d.items()}


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


def map_question_idx_to_attribute_name(idx):
    if idx in category_idx:
        return "category"
    elif idx in color_idx:
        return "color"
    elif idx in brightness_idx:
        return "brightness"
    elif idx in size_idx:
        return "size"
    elif idx in contour_idx:
        return "contour"
    else:
        raise ValueError("The inserted index is wrong")


def generate_data_matrix(n_train,
                         dict_question_to_idx,
                         tuples_train,
                         tuples_test,
                         path_output_folder,
                         path_original_file,
                         h5_file,
                         flag_program=False,
                         test_seen=False):
    """
    This function saves a tuple of the input we need to train and test the
    networks.
    :param n_train: number of examples per question
    :param dict_question_to_idx: dictionary from question string into identifier
    :param tuples_train: tuples containing positive and negative examples at train
    :param tuples_test: tuples containing positive and negative examples at test
    :param path_output_folder: str, where to save the files
    :param path_original_file: str, where the original mnist files are
    :param h5_file: bool, do we want to use h5 file or not
    :param flag_program: if we want to save the programs or not
    :returns: None
    Format of the output file
        (questions, feats, answers)
        three files for each
    It saves in files the two.
    """

    split_name = ["train", "valid", "test"]
    n_list = [n_train, 2000, 2000]
    tuples_list = [tuples_train, tuples_test, tuples_test]

    if test_seen:
        split_name = split_name[1:]
        n_list = n_list[1:]
        tuples_list = tuples_list[1:]

    # if h5_file:
    #     x_split_files = [h5py.File(join(path_output_folder, "feats_%s.hdf5" % s_), 'w')
    #                      for s_ in split_name]
    #    y_split_files = ["y_%s.npy" % s_ for s_ in split_name]

    x_split_files = ["x_%s.npy" % s_ for s_ in split_name]
    y_split_files = ["y_%s.npy" % s_ for s_ in split_name]

    if flag_program:
        raise ValueError("Program generator is still not implemented.")

    # we load the train, validation and test mnist (x,y)
    x_original = [np.load(join(path_original_file, file_)) for file_ in x_split_files]
    y_original = [np.load(join(path_original_file, file_)) for file_ in y_split_files]
    _, dim_x, dim_y = x_original[0].shape  # image dimensions (equivalent across splits)

    n_questions = len(dict_question_to_idx.keys())  # number of questions

    ch_ = 3  # color channel

    for id_, (x_, y_, n_, tuples_, split_) in enumerate(zip(x_original, y_original, n_list, tuples_list, split_name)):

        if h5_file:
            if test_seen:
                f = h5py.File(join(path_output_folder, "feats_in_distr_%s.hdf5" % split_), 'w')
            else:
                f = h5py.File(join(path_output_folder, "feats_%s.hdf5" % split_), 'w')
            dset = f.create_dataset('features',
                                    maxshape=(n_*n_questions, ch_, dim_x, dim_y),
                                    shape=(n_//2, ch_, dim_x, dim_y),
                                    dtype=np.float64,
                                    chunks=True)

        else:
            x_new = np.zeros((n_ * n_questions, ch_, dim_x, dim_y))  # shape n_train, n_valid, etc

        q_new = np.zeros(n_ * n_questions, dtype=int)
        a_new = np.zeros(n_ * n_questions, dtype=int)
        pos_tuples, neg_tuples = tuples_

        digit_indexes = [np.argwhere(y_ == i_).reshape(-1,) for i_ in np.sort(np.unique(y_))]

        count_ = 0
        list_attributes_examples = []
        # question
        for id_tuple, (p_tuples_, q_) in enumerate(zip(pos_tuples, dict_question_to_idx.values())):

            if h5_file:
                reset_count = count_
                x_new = np.zeros((n_//2, ch_, dim_x, dim_y))  # too large
                print('x new shape')
                print(x_new.shape)
            else:
                reset_count = 0

            if split_ == "test":
                np.random.seed(12345)
            n_cat_per_tuple = (n_ // 2) // len(p_tuples_)
            # print("\ncat per tuple", n_cat_per_tuple)
            for p_tuple in p_tuples_:
                images_to_use = np.random.choice(digit_indexes[int(p_tuple[0])], size=n_cat_per_tuple)
                # indexes for the digit in the tuple
                for id_count, id_image in enumerate(images_to_use):
                    if isinstance(p_tuple[4], str):
                        p_tuple[4] = False if p_tuple[4] == 'False' else True
                    # print('tuple')
                    # print(p_tuple)
                    x_new[count_-reset_count] = transform(x_[id_image] / 255,
                                                          reshape=p_tuple[3],
                                                          color=p_tuple[1],
                                                          bright=p_tuple[2],
                                                          contour=p_tuple[4])
                    q_new[count_] = q_
                    a_new[count_] = 1
                    count_ += 1
                    list_attributes_examples.append(p_tuple)

            # print("mod: ", ((n_ // 2) % len(p_tuples_)))
            if ((n_ // 2) % len(p_tuples_)) != 0:
                tmp_list = np.random.choice(np.arange(len(p_tuples_)),
                                            size=(n_ // 2) % len(p_tuples_))
                for id_tmp in tmp_list:
                    tmp_tuple = p_tuples_[id_tmp]
                    id_image = np.random.choice(digit_indexes[int(tmp_tuple[0])])
                    if isinstance(tmp_tuple[4], str):
                        tmp_tuple[4] = False if tmp_tuple[4] == 'False' else True
                    x_new[count_-reset_count] = transform(x_[id_image] / 255,
                                              reshape=tmp_tuple[3],
                                              color=tmp_tuple[1],
                                              bright=tmp_tuple[2],
                                              contour=tmp_tuple[4])
                    list_attributes_examples.append(tmp_tuple)
                    q_new[count_] = q_
                    a_new[count_] = 1
                    count_ += 1
                # images_to_use = np.random.choice(digit_indexes[int(p_tuple[0])], size=n_cat_per_tuple)
            print('count', count_)
            if h5_file:
                if id_tuple > 0:
                    dset.resize(count_, axis=0)
                dset[id_tuple*(n_//2):] = x_new
                print('we are saving')

        print(count_)

        for id_tuple, (n_tuples_, q_) in enumerate(zip(neg_tuples, dict_question_to_idx.values())):
            if h5_file:
                reset_count = count_
                x_new = np.zeros((n_//2, ch_, dim_x, dim_y))
                # else reset is still zero

            n_cat_per_tuple = (n_ // 2) // len(n_tuples_)
            print("cat per tuple", n_cat_per_tuple)
            for n_tuple in n_tuples_:
                images_to_use = np.random.choice(digit_indexes[int(n_tuple[0])], size=n_cat_per_tuple)
                # indexes for the digit in the tuple
                for id_count, id_image in enumerate(images_to_use):
                    if isinstance(n_tuple[4], str):
                        n_tuple[4] = False if n_tuple[4] == 'False' else True
                    x_new[count_-reset_count] = transform(x_[id_image] / 255,
                                              reshape=n_tuple[3],
                                              color=n_tuple[1],
                                              bright=n_tuple[2],
                                              contour=n_tuple[4])
                    list_attributes_examples.append(n_tuple)
                    q_new[count_] = q_
                    a_new[count_] = 0
                    count_ += 1

            print("mod: ", ((n_ // 2) % len(n_tuples_)))
            if ((n_ // 2) % len(n_tuples_)) != 0:
                tmp_list = np.random.choice(np.arange(len(n_tuples_)),
                                            size=(n_ // 2) % len(n_tuples_))
                for id_tmp in tmp_list:
                    tmp_tuple = n_tuples_[id_tmp]
                    id_image = np.random.choice(digit_indexes[int(tmp_tuple[0])])
                    if isinstance(tmp_tuple[4], str):
                        tmp_tuple[4] = False if tmp_tuple[4] == 'False' else True
                    x_new[count_-reset_count] = transform(x_[id_image] / 255,
                                              reshape=tmp_tuple[3],
                                              color=tmp_tuple[1],
                                              bright=tmp_tuple[2],
                                              contour=tmp_tuple[4])
                    list_attributes_examples.append(tmp_tuple)
                    q_new[count_] = q_
                    a_new[count_] = 0
                    count_ += 1

            if h5_file:
                dset.resize(count_, axis=0)
                dset[(n_//2)*n_questions + id_tuple*(n_//2):] = x_new
        print(count_)

        # print("\nCOUNT", count_)
        # print("original n", n_ * n_questions)

        rnd_indexes = np.arange(count_)
        # np.random.seed(123)  # pytorch already load the data in random order
        # np.random.shuffle(rnd_indexes)

        if test_seen:
            split_ = 'in_distr_%s' % split_

        print("SAVE: ", path_output_folder)
        print(x_new.shape)
        if not h5_file:
            np.save(join(path_output_folder, "feats_%s" % split_), x_new[rnd_indexes])
        np.save(join(path_output_folder, "questions_%s" % split_), q_new[rnd_indexes])
        np.save(join(path_output_folder, "answers_%s" % split_), a_new[rnd_indexes])
        np.save(join(path_output_folder, 'attributes_%s' % split_), np.array(list_attributes_examples)[rnd_indexes])

    if not test_seen:
        # save vocab
        with open(join(path_output_folder, 'vocab.json'), 'w') as outfile:
            json.dump(vocab, outfile)


def generate_tuple(output_path,
                   combinations_train_p=1,
                   combinations_train_n=1,
                   combinations_test_p=1,
                   combinations_test_n=1,
                   vocab_=None):
    """
    Function to generate the dataset. We specify the amount of possible combinations
    for the train and test split. In the most general case the dataset has 5
    attributes types. We can specify if we want to use them all or not.
    If these are passed as a list, the list will have the priority,
    If these are passed as a None value, the choice of the attributes is guided
    by the questions in the vocabulary.
    :param output_path:
    :param combinations_train_p: number of tuples that respond positively
    to each question at training
    :param combinations_train_n: number of tuples that respond negatively
    to each question at training
    :param combinations_test_p: number of tuples that respond positively
    to each question at test
    :param combinations_test_n: number of tuples that respond negatively
    to each question at test
    :param vocab_: vocabulary specifying the main parameters for our dataset

    # We save everything in a numpy array, pkl files.
    # :return tuple:
    # :return id_tuple_train_p:
    # :return id_tuple_test_p:
    # :return id_tuple_train_n:
    # :return id_tuple_test_n:
    """
    if vocab_ is None:
        vocab_ = vocab

    attributes_key_dict = {"category": [str(0)],
                           "color": ["red"],
                           "brightness": ["light"],
                           "size": ["large"],
                           "contour": [False]}
    attributes_val_dict = {"%s" % k_: [] for k_ in attributes_key_dict.keys()}

    for val_ in vocab["question_token_to_idx"].values():
        if val_ in category_idx:
            attributes_val_dict["category"].append(val_)
        elif val_ in color_idx:
            attributes_val_dict["color"].append(val_)
        elif val_ in brightness_idx:
            attributes_val_dict["brightness"].append(val_)
        elif val_ in size_idx:
            attributes_val_dict["size"].append(val_)
        elif val_ in contour_idx:
            attributes_val_dict["contour"].append(val_)
        else:
            raise ValueError("Error in the passed keys")

    from_id_to_token = invert_dict(vocab_["question_token_to_idx"])

    for k_attr_family_, list_attr_ in attributes_val_dict.items():
        if len(list_attr_) > 0:
            attributes_key_dict[k_attr_family_] = ([from_id_to_token[a_] for a_ in list_attr_])

    tuples = []
    # generate tuples
    for val1 in attributes_key_dict["category"]:  # str
        for val2 in attributes_key_dict["color"]:  # str
            for val3 in attributes_key_dict["brightness"]:  # str
                for val4 in attributes_key_dict["size"]:  # str
                    for val5 in attributes_key_dict["contour"]:  # bool
                        tuples.append([val1, val2, val3, val4, val5])
    print("tuple size ", len(tuples))
    # train
    output_tuple_train_p = []  # output positive tuples
    id_tuple_train_p = []  # id positive tuples

    output_tuple_train_n = []  # output negative tuples
    id_tuple_train_n = []   # id negative tuples

    # and test
    output_tuple_test_p = []  # output positive tuples
    id_tuple_test_p = []  # id positive tuples

    output_tuple_test_n = []  # output negative tuples
    id_tuple_test_n = []  # id negative tuples

    for k_ in vocab["question_token_to_idx"].keys():  # the key we are looking for

        # TRAIN
        tmp_id_train_p = [id_t_ for id_t_, t_ in enumerate(tuples) if k_ in t_]
        if len(tmp_id_train_p) < combinations_train_p:
            combinations_train_p = len(tmp_id_train_p)
        tmp_id_train_copy_p = tmp_id_train_p.copy()
        new_ids_p = np.random.choice(tmp_id_train_copy_p, size=combinations_train_p, replace=False)
        id_tuple_train_p.append(new_ids_p)
        output_tuple_train_p.append([tuples[i_] for i_ in new_ids_p])

        tmp_id_train_n = [id_t_ for id_t_, t_ in enumerate(tuples) if k_ not in t_]
        if len(tmp_id_train_n) < combinations_train_n:
            combinations_train_n = len(tmp_id_train_n)
        tmp_id_train_copy_n = tmp_id_train_n.copy()
        new_ids_n = np.random.choice(tmp_id_train_copy_n, size=combinations_train_n, replace=False)
        id_tuple_train_n.append(new_ids_n)
        output_tuple_train_n.append([tuples[i_] for i_ in new_ids_n])
        # TEST

        tmp_id_test_copy_p = tmp_id_train_p.copy()
        tmp_id_test_copy_n = tmp_id_train_n.copy()
        try:
            [tmp_id_test_copy_p.remove(j_) for j_ in new_ids_p]
        except:
            print("")

        try:
            [tmp_id_test_copy_n.remove(j_) for j_ in new_ids_n]
        except:
            print("")
        if combinations_test_p == -1:
            new_ids = tmp_id_test_copy_p
        else:
            new_ids = np.random.choice(tmp_id_test_copy_p, size=combinations_test_p, replace=True)
        id_tuple_test_p.append(new_ids)
        output_tuple_test_p.append([tuples[i_] for i_ in new_ids])

        if combinations_test_n == -1:
            new_ids = tmp_id_test_copy_n
        else:
            new_ids = np.random.choice(tmp_id_test_copy_n, size=combinations_test_n, replace=True)
        id_tuple_test_n.append(new_ids)
        output_tuple_test_n.append([tuples[i_] for i_ in new_ids])

    id_tuple_train_n = np.array(id_tuple_train_n)
    id_tuple_train_p = np.array(id_tuple_train_p)

    id_tuple_test_n = np.array(id_tuple_test_n)
    id_tuple_test_p = np.array(id_tuple_test_p)

    np.save(join(output_path, "id_tuple_train_p.npy"), id_tuple_train_p)
    np.save(join(output_path, "id_tuple_train_n.npy"), id_tuple_train_n)
    np.save(join(output_path, "id_tuple_test_p.npy"), id_tuple_test_p)
    np.save(join(output_path, "id_tuple_test_n.npy"), id_tuple_test_n)

    return [output_tuple_train_p, output_tuple_train_n], [output_tuple_test_p, output_tuple_test_n]


def generate_combinations(n_combinations_train=1,
                          n_combinations_test=5,
                          vocab_=None):
    """ Here, we generate the combinations of training and test point.
    We care of keeping some combinations at test across all the experiments.
    Test set is shared across all the experiments which share the same question.
    Deterministic way of generating the data. We start by fixing the amount of
    objects that we will see at training.
    All the attributes must appear at least once.
    :param n_combinations_train: how many repetitions for all the attributes we have at train
    :param  n_combinations_test: how many repetitions for all the attributes we have at test
    :param vocab_: dictionary, global variable
    :return combinations_train: list of lists with all tuples at training
    :return combinations_test: list of lists with all tuples at test
    """
    combinations_train = []  # combinations for training
    combinations_test = []  # combinations for validation and test
    from_id_to_token = invert_dict(vocab_["all_questions_token_to_idx"])

    category_ = []  # list of categories,
    color_ = []  # colors,
    brightness_ = []  # brightness,
    size_ = []  # and sizes from which to extract
    choice_list = [False for k_ in range(4)]  # cat, color, bright, size

    for val_ in vocab["question_token_to_idx"].values():
        if val_ in category_idx:
            category_.append(val_)
            choice_list[0] = True
        elif val_ in color_idx:
            color_.append(val_)
            choice_list[1] = True
        elif val_ in brightness_idx:
            brightness_.append(val_)
            choice_list[2] = True
        elif val_ in size_idx:
            size_.append(val_)
            choice_list[3] = True
        else:
            raise ValueError("Error in the passed keys")

    print("choice list", choice_list)
    # e.g., if None attributes is asked for "color",
    # we enter in the second condition
    if not choice_list[0]:
        category_ = [vocab_["all_questions_token_to_idx"][str(0)]]
    if not choice_list[1]:
        color_ = [vocab_["all_questions_token_to_idx"]["red"]]
    if not choice_list[2]:
        brightness_ = [vocab_["all_questions_token_to_idx"]["light"]]
    if not choice_list[3]:
        size_ = [vocab_["all_questions_token_to_idx"]["large"]]
    # we force the dataset to be red

    # dataset 15-19 random seed 123
    # np.random.seed(123)
    np.random.seed(179)  # in this way the tuples are always the same 20-24
    # np.random.seed(78910)  # 7-12
    # especially at test

    for k_ in range(n_combinations_test):

        generate = True  # no saved tuple
        # we force tuple that appeared at this round not to appear anymore
        # (1, pink, large, bright)
        # (2, red, large, bright)
        # (2, pink, large, bright) is not allowed
        # through dict_control
        dict_control = {k_: False for k_ in vocab["question_token_to_idx"].keys()}

        while generate:
            tmp_tuple = [np.random.choice(category_),
                         np.random.choice(color_),
                         np.random.choice(brightness_),
                         np.random.choice(size_)]  # random tuple

            tmp_check = True  # the tuple already exists
            str_tuple = []  # how the tuple translates into string
            for id_tmp_, tmp_ in enumerate(tmp_tuple):
                tmp_str = from_id_to_token[tmp_]  # we translate into string
                str_tuple.append(tmp_str)  #  and we append in the tuple
                if choice_list[id_tmp_]:  # if we have freedom on that parameter, multiply
                    tmp_check *= dict_control[tmp_str]
            exists = str_tuple in combinations_test
            if not tmp_check and not exists:  # if the tuple does not exist already
                combinations_test.append(str_tuple)  # no contour : enforced
                for tmp_str in str_tuple:
                    dict_control[tmp_str] = True
            generate = not reduce(lambda x, y: x * y, list(dict_control.values()))

    for k_ in range(n_combinations_train):

        generate = True
        dict_control = {k_: False for k_ in vocab["question_token_to_idx"].keys()}

        while generate:

            tmp_tuple = [np.random.choice(category_),
                         np.random.choice(color_),
                         np.random.choice(brightness_),
                         np.random.choice(size_)]

            tmp_check = True  # the tuple already exists
            str_tuple = []  # how the tuple translates into string
            for id_tmp_, tmp_ in enumerate(tmp_tuple):
                tmp_str = from_id_to_token[tmp_]  # we translate into string
                str_tuple.append(tmp_str)  #  and we append in the tuple
                if choice_list[id_tmp_]:  # if we have freedom on that parameter, multiply
                    tmp_check *= dict_control[tmp_str]
            exists = str_tuple in combinations_test or str_tuple in combinations_train
            if not tmp_check and not exists:  # if the tuple does not exist already
                combinations_train.append(str_tuple)
                for tmp_str in str_tuple:
                    dict_control[tmp_str] = True
            generate = not reduce(lambda x, y: x * y, list(dict_control.values()))

    [c_.append(False) for c_ in combinations_train]
    [c_.append(False) for c_ in combinations_test]

    return combinations_train, combinations_test


def split_combinations(train_combinations,
                       test_combinations,
                       force_balance=False):
    """ We split the combinations based on the question, into positive and negative.
    :param train_combinations: all the combinations we will see at training
    :param test_combinations: all the combinations that we see at test
    :param force_balance: if force_balance, same #tuples for train and test
    :returns [train_p_tuples, train_n_tuples], [test_p_tuples, test_n_tuples]:
        train_p_tuples, list of positive tuples at training
        train_n_tuples, list of negative tuples at training
        test_p_tuples, list of positive tuples at test
        test_n_tuples, list of negative tuples at test
     """
    n_train_comb = len(train_combinations)
    n_test_comb = len(test_combinations)
    train_p_tuples = []
    train_n_tuples = []
    test_p_tuples = []
    test_n_tuples = []

    for qk_, qv_ in vocab["question_token_to_idx"].items():
        idx_tuple = map_question_idx_to_attribute_category(qv_)
        id_tr_p = np.argwhere([tpl[idx_tuple] == qk_ for tpl in train_combinations]).squeeze()
        id_ts_p = np.argwhere([tpl[idx_tuple] == qk_ for tpl in test_combinations]).squeeze()
        if id_tr_p.size == 1:
            id_tr_p = [int(id_tr_p)]
        if id_ts_p.size == 1:
            id_ts_p = [int(id_ts_p)]

        lst_id_tr = len(id_tr_p) if force_balance else -1
        lst_id_ts = len(id_ts_p) if force_balance else -1

        train_p_tuples.append([train_combinations[id_] for id_ in id_tr_p])
        test_p_tuples.append([test_combinations[id_] for id_ in id_ts_p])
        train_n_tuples.append([train_combinations[id_]
                               for id_ in np.delete(np.arange(n_train_comb), id_tr_p)[:lst_id_tr]])
        test_n_tuples.append([test_combinations[id_]
                              for id_ in np.delete(np.arange(n_test_comb), id_ts_p)[:lst_id_ts]])

    return [train_p_tuples, train_n_tuples], [test_p_tuples, test_n_tuples]


def generate_data_file(output_data_folder,
                       n_train,
                       n_tuples_train_p,
                       n_tuples_train_n,
                       n_tuples_test_p,
                       n_tuples_test_n,
                       splits_folder,
                       h5_file,
                       random_combinations=False,
                       test_seen=False,
                       dataset_name=None
                       ):


    # TODO: RANDOM COMBINATION
    # ALWAYS SET TO FALSE
    if test_seen:
        if dataset_name is None:
            raise ValueError("Provide dataset_name")
        dataset_path_ = join(output_data_folder, dataset_name)
        attr_tr = np.load(join(dataset_path_, 'attributes_train.npy'))
        tmp_comb = np.unique(attr_tr, axis=0)
        comb_train = [[j__ for j__ in j_] for j_ in tmp_comb]
        comb_test = comb_train
        print(comb_train)
        print("\n\n")
        print(comb_test)

    else:
        info = {}
        info_path = output_data_folder + 'data_list.json'
        print(info_path)
        dirname = os.path.dirname(info_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            idx_base = 0
        elif os.path.isfile(info_path):
            with open(info_path) as infile:
                info = json.load(infile)
                if info:  # it is not empty
                    idx_base = int(list(info.keys())[-1]) + 1  # concatenate
                else:
                    idx_base = 0
        else:  # we have the folder but the file is empty
            info = dict()
            idx_base = 0

        if not dataset_name is None:
            index = int(dataset_name.split("_")[-1])
            if index >= idx_base:
                idx_base = index
        print('dataset id', idx_base)

        os.makedirs(output_data_folder + "dataset_%i" % idx_base,
                    exist_ok=False)

        if random_combinations:  # old code
            tuples_train, tuples_test = generate_tuple(output_path=output_data_folder + "dataset_%i" % idx_base,
                                                       combinations_train_p=n_tuples_train_p,
                                                       combinations_train_n=n_tuples_train_n,
                                                       combinations_test_p=n_tuples_test_p,
                                                       combinations_test_n=n_tuples_test_n,
                                                       vocab_=None)
        # print(len(tuples_train), len(tuples_test))  it is divided in training and test
        # test tuples are typically not balanced
        # print(len(tuples_test[0][0]), len(tuples_test[1][0]))  # positive and negative

        else:
            comb_train, comb_test = generate_combinations(n_combinations_train=n_tuples_train_p,
                                                          vocab_=vocab)

    tuples_train, tuples_test = split_combinations(comb_train,
                                                   comb_test,
                                                   force_balance=False)

    if not test_seen:

        dataset_path_ = output_data_folder + "dataset_%i" % idx_base

        info[idx_base] = {"dataset_name": "dataset_%i" % idx_base,
                          "dataset_path": output_data_folder + "dataset_%i" % idx_base,
                          "dataset_train": tuples_train,
                          "dataset_test": tuples_test,
                          "question_token": vocab["question_token_to_idx"],
                          "positive_combinations_train": n_tuples_train_p,
                          "negative_combinations_train": n_tuples_train_n,
                          "positive_combinations_test": n_tuples_test_p,
                          "negative_combinations_test": n_tuples_test_n}

        with open(info_path, 'w') as f:
            json.dump(info, f)

    generate_data_matrix(n_train,
                         vocab["question_token_to_idx"],
                         tuples_train,
                         tuples_test,
                         h5_file=h5_file,
                         path_output_folder=dataset_path_,
                         path_original_file=splits_folder,
                         flag_program=False,
                         test_seen=test_seen
                         )
    return


def build_mapping(path_data, on_seen=False):
    split_list = ["train", "valid", "test"]
    for split_ in split_list:
        if on_seen and split_ == 'train':
            continue
        elif on_seen:
            split_ = "in_distr_%s" % split_
        attributes = np.load(join(path_data, "attributes_%s.npy" % split_))
        dct = {}
        question_ = np.array([], dtype=int)
        n, a = attributes.shape
        for k_ in range(attributes.shape[1]-1):  # contour not included
            unique_attributes = np.sort(np.unique(attributes[:, k_]))
            dct.update({attr: j for j, attr in enumerate(unique_attributes)})
            if unique_attributes.size > 1:
                question_ = np.append(question_, k_)
        answers_array = np.zeros((n, question_.size), dtype=int)
        questions_array = np.ones((n, question_.size), dtype=int) * question_
        for i_, attr_sample in enumerate(attributes[:, question_]):
            for j_, attr_val in enumerate(attr_sample):
                answers_array[i_, j_] = dct[attr_val]

        np.save(join(path_data, "questions_query_%s.npy" % split_), questions_array)
        np.save(join(path_data, "answers_query_%s.npy" % split_), answers_array)

    return


def build_out_of_sample_test(path_data, splits_folder):
    """ This is an emergency function.
    We define it because the tuples at validation and test are equivalent,
    we may see over-fitting.
    :param path_data: path to the dataset for VQA
    :param splits_folder: path to the MNIST splits
    """
    # these list must be changed depending on the choice you made
    vocab = json.load(open(join(path_data, "vocab.json"), 'rb'))
    category_list = np.arange(10)
    color_list = ["red", "green", "blue", "yellow", "purple"]
    bright_list = ["dark", "half", "light"]
    size_list = ["small", "medium", "large"]

    all_tuples = []
    for category in category_list:
        for color in color_list:
            for bright in bright_list:
                for size in size_list:
                    all_tuples.append([str(category), color, bright, size, "False"])

    # you want to load the data that contains the largest number of categories
    # it can be used for the smaller training case
    attr_tr = np.unique(np.load(join(path_data, "attributes_train.npy")), axis=0)
    attr_vl = np.unique(np.load(join(path_data, "attributes_valid.npy")), axis=0)
    attr_ts = np.unique(np.load(join(path_data, "attributes_test.npy")), axis=0)

    if np.array_equal(attr_vl, attr_ts):
        # validation and test tuples are equivalent
        count_new = 0
        left_tuples = []

        for t_ in all_tuples:  # for every tuple
            exists = False
            for ttrain_ in attr_tr:  # we check all the tuples at training
                if all([t__ == ttrain__ for t__, ttrain__ in zip(t_, ttrain_)]):
                    exists = True

            for ttest_ in attr_ts:  # for all the tuples at test
                if all([t__ == ttest__ for t__, ttest__ in zip(t_, ttest_)]):
                    exists = True

            if not exists:
                count_new += 1
                left_tuples.append(t_)

    else:
        raise ValueError("Validation split needs to be taken into account, \noption not implemented")

    n_test_comb = len(left_tuples)  # for all the available test combinations
    # specify # data points per question
    pos_tuples = []
    neg_tuples = []
    for qk_, qv_ in vocab["question_token_to_idx"].items():
        idx_tuple = map_question_idx_to_attribute_category(qv_)
        id_ts_p = np.argwhere([tpl[idx_tuple] == qk_ for tpl in left_tuples]).squeeze()
        if id_ts_p.size == 1:
            id_ts_p = [int(id_ts_p)]

        pos_tuples.append([left_tuples[id_] for id_ in id_ts_p])
        neg_tuples.append([left_tuples[id_] for id_ in np.delete(np.arange(n_test_comb), id_ts_p)])

    flag_skip = np.zeros(len(vocab["question_token_to_idx"].keys()), dtype=int)
    for id_, (p_, n_, qk_) in enumerate(zip(pos_tuples, neg_tuples, vocab["question_token_to_idx"].keys())):
        flag_skip[id_] = (len(p_) == 0 or len(n_) == 0)
        print("%s. Pos: %i. Neg: %i." % (qk_, len(p_), len(n_)))

    # return [test_p_tuples, test_n_tuples]
    split_ = "test"
    x_ = np.load(join(splits_folder, "x_test.npy")) / 255
    y_ = np.load(join(splits_folder, "y_test.npy"))
    print("\nMax: ", np.max(x_))
    # we load the train, validation and test mnist (x,y)
    dim_x, dim_y = x_[0].shape  # image dimensions (equivalent across splits)

    n_questions = len(vocab["question_token_to_idx"].keys())  # number of questions
    n_ = 2000
    ch_ = 3  # color channel

    # for id_, (x_, y_, n_, tuples_, split_) in enumerate(zip(x_original, y_original, n_list, tuples_list, split_name)):
    print("\n%s" % split_)
    x_new = np.zeros((n_ * n_questions, ch_, dim_x, dim_y))
    q_new = np.zeros(n_ * n_questions, dtype=int)
    a_new = np.zeros(n_ * n_questions, dtype=int)

    digit_indexes = [np.argwhere(y_ == i_).reshape(-1, ) for i_ in np.sort(np.unique(y_))]

    count_ = 0
    list_attributes_examples = []
    # question
    for id_tuple, (p_tuples_, q_) in enumerate(zip(pos_tuples, vocab["question_token_to_idx"].values())):
        if split_ == "test":
            np.random.seed(12345)
        if flag_skip[id_tuple]:
            continue
        n_cat_per_tuple = (n_ // 2) // len(p_tuples_)
        print("\ncat per tuple", n_cat_per_tuple)
        for p_tuple in p_tuples_:
            images_to_use = np.random.choice(digit_indexes[int(p_tuple[0])], size=n_cat_per_tuple)
            # indexes for the digit in the tuple
            for id_count, id_image in enumerate(images_to_use):
                contour_flag = False if p_tuple[4] == "False" else True
                x_new[count_] = transform(x_[id_image],
                                          reshape=p_tuple[3],
                                          color=p_tuple[1],
                                          bright=p_tuple[2],
                                          contour=contour_flag)
                q_new[count_] = q_
                a_new[count_] = 1
                count_ += 1
                list_attributes_examples.append(p_tuple)

        print("mod: ", ((n_ // 2) % len(p_tuples_)))
        if ((n_ // 2) % len(p_tuples_)) != 0:
            tmp_list = np.random.choice(np.arange(len(p_tuples_)),
                                        size=(n_ // 2) % len(p_tuples_))
            for id_tmp in tmp_list:
                tmp_tuple = p_tuples_[id_tmp]
                id_image = np.random.choice(digit_indexes[int(tmp_tuple[0])])
                contour_flag = False if tmp_tuple[4] == "False" else True
                x_new[count_] = transform(x_[id_image],
                                          reshape=tmp_tuple[3],
                                          color=tmp_tuple[1],
                                          bright=tmp_tuple[2],
                                          contour=contour_flag)
                list_attributes_examples.append(tmp_tuple)
                q_new[count_] = q_
                a_new[count_] = 1
                count_ += 1
            # images_to_use = np.random.choice(digit_indexes[int(p_tuple[0])], size=n_cat_per_tuple)
    print(count_)

    for id_tuple, (n_tuples_, q_) in enumerate(zip(neg_tuples, vocab["question_token_to_idx"].values())):
        if flag_skip[id_tuple]:
            continue
        n_cat_per_tuple = (n_ // 2) // len(n_tuples_)
        print("cat per tuple", n_cat_per_tuple)
        for n_tuple in n_tuples_:
            images_to_use = np.random.choice(digit_indexes[int(n_tuple[0])], size=n_cat_per_tuple)
            # indexes for the digit in the tuple
            for id_count, id_image in enumerate(images_to_use):
                contour_flag = False if n_tuple[4] == "False" else True
                x_new[count_] = transform(x_[id_image],
                                          reshape=n_tuple[3],
                                          color=n_tuple[1],
                                          bright=n_tuple[2],
                                          contour=contour_flag)
                list_attributes_examples.append(n_tuple)
                q_new[count_] = q_
                a_new[count_] = 0
                count_ += 1

        print("mod: ", ((n_ // 2) % len(n_tuples_)))
        if ((n_ // 2) % len(n_tuples_)) != 0:
            tmp_list = np.random.choice(np.arange(len(n_tuples_)),
                                        size=(n_ // 2) % len(n_tuples_))
            for id_tmp in tmp_list:
                tmp_tuple = n_tuples_[id_tmp]
                id_image = np.random.choice(digit_indexes[int(tmp_tuple[0])])
                contour_flag = False if tmp_tuple[4] == "False" else True
                x_new[count_] = transform(x_[id_image],
                                          reshape=tmp_tuple[3],
                                          color=tmp_tuple[1],
                                          bright=tmp_tuple[2],
                                          contour=contour_flag)
                list_attributes_examples.append(tmp_tuple)
                q_new[count_] = q_
                a_new[count_] = 0
                count_ += 1
    print(count_)

    rnd_indexes = np.arange(count_)
    np.random.seed(123)
    np.random.shuffle(rnd_indexes)
    print("x_new max")
    np.save(join(path_data, "feats_oos_%s" % split_), x_new[rnd_indexes])
    np.save(join(path_data, "questions_oos_%s" % split_), q_new[rnd_indexes])
    np.save(join(path_data, "answers_oos_%s" % split_), a_new[rnd_indexes])
    np.save(join(path_data, 'attributes_oos_%s' % split_), np.array(list_attributes_examples)[rnd_indexes])


# TODO: generate dense VQA questions
def gen_dense_questions(path_dataset):
    """ Starting from the attributes we generate a random set of questions.
    Each question per attribute type, to be comparable to the query type of thing.
    We have to save an array of dense_questions.npy and dense_answers.npy
    """
    splits = ["train", "valid", "test", "oos_test"]
    vocab_ = json.load(open(join(path_dataset, "vocab.json"), "rb"))["question_token_to_idx"]

    for split_ in splits:
        attributes_filename = "attributes_%s.npy" % split_
        tmp_ = np.load(join(path_dataset, attributes_filename))[:, :-1]  # we take the original attributes
        dense_questions = tmp_.copy()  # we save the dense questions
        key_ = "change"
        n_, a_ = tmp_.shape
        dense_y = np.random.choice(2, size=(n_, a_))  #  we generate the dense labels
        # if y == 0 we need to change the attribute into a new question
        new_ = np.where(dense_y, dense_questions, key_)
        dense_questions_final = np.zeros((n_, a_), dtype=int)
        for k_ in range(a_):
            existing_attr = np.array([v_ for v_ in np.unique(dense_questions[:, k_]) if v_ != key_])
            bm_change = new_[:, k_] == key_
            mis_attr = np.array([np.random.choice(np.delete(existing_attr, np.argwhere(existing_attr == a__)))
                                 for a__ in tmp_[bm_change, k_]])
            dense_questions[bm_change, k_] = mis_attr

        for i, sample in enumerate(dense_questions):
            for j, s_ in enumerate(sample):
                dense_questions_final[i, j] = vocab_[s_]

        np.save(join(path_dataset, "dense_questions_%s.npy" % split_), dense_questions_final)
        np.save(join(path_dataset, "dense_answers_%s.npy" % split_), dense_y)
