import os
import sys
import numpy as np
import pandas as pd
import json
import pickle
from os.path import join
from functools import reduce
from runs.mnist_attribute_generation import transform


# TODO: GENERATE VOCABULARY
# questions -- category(0-9), color(10-14), brightness(15-17), size(18-20), contour(21)
# category          0-9
# color             10: red, 11: green, 12: blue, 13: yellow, 14: purple
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

vocab = {}
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
    combinations_test = []   # combinations for validation and test
    from_id_to_token = invert_dict(vocab_["all_questions_token_to_idx"])

    category_ = []    # list of categories,
    color_ = []       # colors,
    brightness_ = []  # brightness,
    size_ = []        # and sizes from which to extract
    choice_list = [False for k_ in range(4)]  # cat, color, bright, size

    for val_ in vocab["question_token_to_idx"].values():
        print(val_, type(val_))
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
    np.random.seed(123)  # in this way the tuples are always the same
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
    """
    We split the combinations based on the question, into positive and negative.
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


class ImageGenerator(object):
    def __init__(self,
                 all_images,
                 image_size=64):
        """
        :param all_images: numpy matrix containing images
        :param output_image_size: output image dimension
        """
        self.all_images = all_images
        self.image_size = image_size
        _, dimx, dimy = all_images.shape
        if dimx == dimy:
            self.digit_size = dimx
        else:
            raise ValueError("These are not square images.")

    def _set_positions(self):
        flag = True
        while flag:
            pos1 = [np.random.choice(np.arange(self.image_size - self.digit_size)),
                    np.random.choice(np.arange(self.image_size - self.digit_size))]
            pos2 = [np.random.choice(np.arange(self.image_size - self.digit_size)),
                    np.random.choice(np.arange(self.image_size - self.digit_size))]
            flag = (abs(pos1[0] - pos2[0]) < self.digit_size and abs(pos1[1] - pos2[1]) < self.digit_size)
        return pos1, pos2

    def generate_img(self, tuple1, tuple2, id1, id2):
        #
        n_ch = 3
        img = np.zeros((n_ch, self.image_size, self.image_size))
        pos1, pos2 = self._set_positions()
        x_new1 = transform(self.all_images[id1] / 255,
                           reshape=tuple1[3],
                           color=tuple1[1],
                           bright=tuple1[2],
                           contour=tuple1[4])

        x_new2 = transform(self.all_images[id2] / 255,
                           reshape=tuple2[3],
                           color=tuple2[1],
                           bright=tuple2[2],
                           contour=tuple2[4])

        yy, xx = np.meshgrid(np.arange(pos1[0], pos1[0] + self.digit_size),
                             np.arange(pos1[1], pos1[1] + self.digit_size))
        img[:, xx, yy] = x_new1
        yy, xx = np.meshgrid(np.arange(pos2[0], pos2[0] + self.digit_size),
                             np.arange(pos2[1], pos2[1] + self.digit_size))
        img[:, xx, yy] = x_new2

        return img


def generate_data_matrix(n_train,
                         dict_question_to_idx,
                         tuples_train,
                         tuples_test,
                         path_output_folder,
                         path_original_file,
                         flag_program=False):
    """
    This function saves a tuple of the input we need to train and test the
    networks.
    :param n_train: number of examples per question
    :param dict_question_to_idx: dictionary from question string into identifier
    :param tuples_train: tuples containing positive and negative examples at train
    :param tuples_test: tuples containing positive and negative examples at test
    :param path_output_folder: str, where to save the files
    :param path_original_file: str, where the original mnist files are
    :param flag_program: if we want to save the programs or not
    :returns: None
    Format of the output file
        (questions, feats, answers)
        three files for each
    It saves in files the two.
    """
    split_name = ["train", "valid", "test"]
    n_questions = len(vocab["question_token_to_idx"].keys())  # number of questions
    n_list = [n_train, 2000, 2000]
    ch_ = 3  # color channel
    img_size = 64

    for id_, (split_, n_) in enumerate(zip(split_name, n_list)):
        x_ = np.load(join(path_original_file, "x_%s.npy" % split_))
        y_ = np.load(join(path_original_file, "y_%s.npy" % split_))
        IG = ImageGenerator(all_images=x_)
        dim_x, dim_y = x_[0].shape  # image dimensions (equivalent across splits)
        if split_ == "train":
            tuples_ = tuples_train
        else:
            tuples_ = tuples_test

        x_new = np.zeros((n_ * n_questions, ch_, img_size, img_size))
        q_new = np.zeros(n_ * n_questions, dtype=int)
        a_new = np.zeros(n_ * n_questions, dtype=int)
        pos_tuples, neg_tuples = tuples_

        digit_indexes = [np.argwhere(y_ == i_).reshape(-1, ) for i_ in np.sort(np.unique(y_))]

        count_ = 0
        list_attributes_examples = []
        # question

        # positive
        for id_tuple, (p_tuples_, q_) in enumerate(zip(pos_tuples, dict_question_to_idx.values())):
            if split_ == "test":
                np.random.seed(12345)
            n_cat_per_tuple = (n_ // 2) // len(p_tuples_)  # n positive answers for q_
            neg_tuple_per_q = neg_tuples[id_tuple]  # negative tuples for q_
            el_n_tuples_ = len(neg_tuple_per_q)  # number of neg tuples

            for p_tuple in p_tuples_:  # for every tuple that provide positive answer
                # positive elements: we pick their indexes from mnist
                images_to_use_p = np.random.choice(digit_indexes[int(p_tuple[0])], size=n_cat_per_tuple)
                # confounder: we fix the tuples
                elem_2_tuple = [neg_tuple_per_q[jjj_] for jjj_ in
                                np.random.choice(np.arange(el_n_tuples_),
                                                 size=n_cat_per_tuple)]
                # negative elements: we pick their indexes from mnist
                images_to_use_n = np.array([np.random.choice(digit_indexes[int(el__[0])], size=1)
                                            for el__ in elem_2_tuple]).flatten()
                # image generation
                for id_count, (id_img_p, id_img_n, elem_2) in enumerate(zip(images_to_use_p,
                                                                            images_to_use_n,
                                                                            elem_2_tuple)):
                    x_new[count_] = IG.generate_img(p_tuple, elem_2, id_img_p, id_img_n)
                    q_new[count_] = q_
                    a_new[count_] = 1
                    count_ += 1
                    list_attributes_examples.append([p_tuple, elem_2])

            modulo_datapoints = (n_ // 2) % len(p_tuples_)
            # in case we have still some datapoint to generate
            if modulo_datapoints != 0:
                # list for the positive and neg element
                tmp_list_1 = np.random.choice(np.arange(len(p_tuples_)),
                                              size=modulo_datapoints)
                tmp_list_2 = np.random.choice(np.arange(el_n_tuples_),
                                              size=modulo_datapoints)

                for id_tmp_1, id_tmp_2, elem_2 in zip(tmp_list_1, tmp_list_2, elem_2_tuple):
                    tmp_tuple_1 = p_tuples_[id_tmp_1]  # positive element
                    # tmp_tuple_2 = neg_tuples[id_tuple][id_tmp_2]  # negative element
                    tmp_tuple_2 = neg_tuple_per_q[id_tmp_2]
                    id_image_1 = np.random.choice(digit_indexes[int(tmp_tuple_1[0])])
                    id_image_2 = np.random.choice(digit_indexes[int(tmp_tuple_2[0])])
                    x_new[count_] = IG.generate_img(tmp_tuple_1, tmp_tuple_2,
                                                    id_image_1, id_image_2)
                    q_new[count_] = q_
                    a_new[count_] = 1
                    list_attributes_examples.append([tmp_tuple_1, tmp_tuple_2])

                    count_ += 1

        for id_tuple, (n_tuples_, q_) in enumerate(zip(neg_tuples, dict_question_to_idx.values())):
            n_cat_per_tuple = (n_ // 2) // len(n_tuples_)
            for n_tuple in n_tuples_:
                images_to_use_1 = np.random.choice(digit_indexes[int(n_tuple[0])],
                                                   size=n_cat_per_tuple)

                n_tuples_copy = n_tuples_.copy()  # we do not want two copies of the
                n_tuples_copy.remove(n_tuple)  # same objects
                elem_2_tuple = [n_tuples_copy[jjj_] for jjj_
                                in np.random.choice(np.arange(len(n_tuples_copy)),
                                                    size=n_cat_per_tuple)]
                images_to_use_2 = np.array([np.random.choice(digit_indexes[int(el__[0])], size=1)
                                            for el__ in elem_2_tuple]).flatten()

                # indexes for the digit in the tuple
                for id_image_1, id_image_2, elem2 in zip(images_to_use_1, images_to_use_2, elem_2_tuple):
                    x_new[count_] = IG.generate_img(n_tuple, elem2, id_image_1, id_image_2)
                    q_new[count_] = q_
                    a_new[count_] = 0
                    count_ += 1
                    list_attributes_examples.append([n_tuple, elem2])

            modulo_datapoints = (n_ // 2) % len(n_tuples_)
            if modulo_datapoints != 0:
                tmp_list_1 = np.random.choice(np.arange(len(n_tuples_)),
                                              size=modulo_datapoints)
                tmp_list_2 = [np.random.choice(np.delete(np.arange(len(n_tuples_)),
                                                         tmp_list_1[kkk_]))
                              for kkk_ in range(modulo_datapoints)]

                for id_tmp_1, id_tmp_2 in zip(tmp_list_1, tmp_list_2):
                    tmp_tuple_1 = n_tuples_[id_tmp_1]
                    tmp_tuple_2 = n_tuples_[id_tmp_2]
                    id_image_1 = np.random.choice(digit_indexes[int(tmp_tuple_1[0])])
                    id_image_2 = np.random.choice(digit_indexes[int(tmp_tuple_2[0])])

                    x_new[count_] = IG.generate_img(tmp_tuple_1, tmp_tuple_2, id_image_1, id_image_2)
                    list_attributes_examples.append([tmp_tuple_1, tmp_tuple_2])
                    q_new[count_] = q_
                    a_new[count_] = 0
                    count_ += 1
        print(count_)

        rnd_indexes = np.arange(count_)
        np.random.seed(123)
        np.random.shuffle(rnd_indexes)

        np.save(join(path_output_folder, "feats_%s" % split_), x_new[rnd_indexes])
        np.save(join(path_output_folder, "questions_%s" % split_), q_new[rnd_indexes])
        np.save(join(path_output_folder, "answers_%s" % split_), a_new[rnd_indexes])
        np.save(join(path_output_folder, 'attributes_%s' % split_), np.array(list_attributes_examples)[rnd_indexes])

    # save vocab
    with open(join(path_output_folder, 'vocab.json'), 'w') as outfile:
        json.dump(vocab, outfile)


def generate_data_file(output_data_folder,
                       n_train,
                       n_tuples_train,
                       n_tuples_test,
                       splits_folder):
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

    os.makedirs(output_data_folder + "dataset_%i" % idx_base,
                exist_ok=False)

    comb_train, comb_test = generate_combinations(n_combinations_train=n_tuples_train,
                                                  n_combinations_test=n_tuples_test,
                                                  vocab_=vocab)
    print("\nTrain combinations")
    print(comb_train)
    print("\nTest combinations")
    print(comb_test)

    # this is based on question -- we want to sample uniformly
    # from the square of possible combinations
    tuples_train, tuples_test = split_combinations(comb_train,
                                                   comb_test,
                                                   force_balance=False)
    print("\nTuples train")
    for q_, t_ in zip(vocab["question_token_to_idx"].keys(), tuples_train[1]):
        print("\n" + q_)
        print(t_)
    # return
    dataset_path_ = output_data_folder + "dataset_%i" % idx_base

    info[idx_base] = {"dataset_name": "dataset_%i" % idx_base,
                      "dataset_path": output_data_folder + "dataset_%i" % idx_base,
                      "dataset_train": tuples_train,
                      "dataset_test": tuples_test,
                      "question_token": vocab["question_token_to_idx"],
                      "combinations_train": n_tuples_train,
                      "combinations_test": n_tuples_test}

    with open(info_path, 'w') as f:
        json.dump(info, f)

    generate_data_matrix(n_train,
                         vocab["question_token_to_idx"],
                         tuples_train,
                         tuples_test,
                         path_output_folder=dataset_path_,
                         path_original_file=splits_folder,
                         flag_program=False)