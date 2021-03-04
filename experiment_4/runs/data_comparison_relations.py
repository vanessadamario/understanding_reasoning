import os
import h5py
import json
import numpy as np
from PIL import Image
from functools import reduce
from os.path import join
import pickle


MNIST_DIGITS = [str(k_) for k_ in np.arange(10)]
color_dict = {'red': [1, 0, 0],
              'green': [0, 1, 0],
              'blue': [0, 0, 1],
              'yellow': [1, 1, 0],
              'purple': [1, 0, 1]}
resize_dict = {'small': 14,
               'medium': 20,
               'large': 28}
brightness_dict = {'dark': 0.4,
                   'half': 0.7,
                   'light': 1.}

ATTRIBUTES = (MNIST_DIGITS +
              list(color_dict.keys()) +
              list(brightness_dict.keys()) +
              list(resize_dict.keys()))
RELATIONS = ["same_size", "different_size", "smaller", "larger",
             "same_color", "different_color",
             "same_shape", "different_shape",
             "same_luminance", "different_luminance", "brighter", "darker",
             "left_of", "right_of", "above", "below"
             ]

ATTRIBUTES_DICT = {k_: i_ for i_, k_ in enumerate(ATTRIBUTES)}
RELATIONS_DICT = {k_: i_+len(ATTRIBUTES) for i_, k_ in enumerate(RELATIONS)}
RELATIONS_DICT_EXP2 = {k_: i_+len(ATTRIBUTES) for i_, k_ in enumerate(RELATIONS[:-4])}
RELATIONS_SPATIAL_ONLY = {k_: i_+len(ATTRIBUTES) for i_, k_ in enumerate(RELATIONS[-4:])}

# vocab = {}
# vocab["all_questions_token_to_idx"] = ATTRIBUTES_DICT.copy()
# vocab["all_questions_token_to_idx"].update(RELATIONS_DICT.copy())
# vocab["question_token_to_idx"] = vocab["all_questions_token_to_idx"].copy()  # FIXME

# vocab["answer_token_to_idx"] = {"true": 1, "false": 0}

category_idx = [k_ for k_ in range(10)]
color_idx = [k_ for k_ in range(10, 15)]
brightness_idx = [k_ for k_ in range(15, 18)]
size_idx = [k_ for k_ in range(18, 21)]

# q_size_idx = [k_ for k_ in range(21, 25)]
# q_color_idx = [k_ for k_ in range(25, 27)]
# q_category_idx = [k_ for k_ in range(27, 29)]
# q_brightness_idx = [k_ for k_ in range(29, 33)]
# q_spatial_rel_idx = [k_ for k_ in range(33, 37)]


def generate_vocab(single_image=False, spatial_only=False):
    """ Experiment type.
    If single_image, spatial_only is false
    spatial_only
    single_image
    """
    vocab = {}

    vocab["all_questions_token_to_idx"] = ATTRIBUTES_DICT.copy()
    if (not single_image) and spatial_only:
        raise ValueError('Impossible spatial relation between separated objects.')
    if not single_image:
        vocab["all_questions_token_to_idx"].update(RELATIONS_DICT_EXP2.copy())
    elif spatial_only:
        vocab["all_questions_token_to_idx"].update(RELATIONS_SPATIAL_ONLY.copy())
    else:
        vocab["all_questions_token_to_idx"].update(RELATIONS_DICT.copy())

    vocab["question_token_to_idx"] = vocab["all_questions_token_to_idx"].copy()  # FIXME
    vocab["answer_token_to_idx"] = {"true": 1, "false": 0}

    return vocab


def generate_list(single_image=False, spatial_only=False):
    if (not single_image) and spatial_only:
        raise ValueError('Impossible spatial relation between separated objects.')
    count = len(category_idx) + len(color_idx) + len(brightness_idx) + len(size_idx)

    if spatial_only:
        q_spatial_rel_idx = [k_ for k_ in range(count, count + len(RELATIONS_SPATIAL_ONLY))]
        return q_spatial_rel_idx

    elif not single_image:
        q_size_idx = [k_ for k_ in range(4, 8)]
        q_color_idx = [k_ for k_ in range(8, 10)]
        q_category_idx = [k_ for k_ in range(10, 12)]
        q_brightness_idx = [k_ for k_ in range(12, 16)]
        return q_size_idx, q_color_idx, q_category_idx, q_brightness_idx
    else:
        q_size_idx = [k_ for k_ in range(21, 25)]
        q_color_idx = [k_ for k_ in range(25, 27)]
        q_category_idx = [k_ for k_ in range(27, 29)]
        q_brightness_idx = [k_ for k_ in range(29, 33)]
        q_spatial_rel_idx = [k_ for k_ in range(33, 37)]
        return q_size_idx, q_color_idx, q_category_idx, q_brightness_idx, q_spatial_rel_idx


def map_question_idx_to_group_all(idx, single_image=True):
    """ Return the module id - where attributes,
    comparisons and relational questions are collected
    in subgroups.
    :param idx: identifier for the question
    """
    if single_image:
        q_size_idx, q_color_idx, q_category_idx, q_brightness_idx, q_spatial_rel_idx = generate_list(single_image=True,
                                                                                                     spatial_only=False)
        if idx in category_idx:
            return 0
        elif idx in color_idx:
            return 1
        elif idx in brightness_idx:
            return 2
        elif idx in size_idx:
            return 3
    else:
        q_size_idx, q_color_idx, q_category_idx, q_brightness_idx = generate_list(single_image=False,
                                                                                  spatial_only=False)
        if idx < 4:
            return idx
        q_spatial_rel_idx = []

    if idx in q_size_idx:
        return 4
    elif idx in q_color_idx:
        return 5
    elif idx in q_category_idx:
        return 6
    elif idx in q_brightness_idx:
        return 7
    elif idx in q_spatial_rel_idx:
        return 8
    else:
        raise ValueError("The inserted index is wrong")


def map_question_idx_to_group_spatial_only(idx):
    q_spatial_rel_idx = generate_list(single_image=True,
                                      spatial_only=True)
    if idx in category_idx:
        return 0
    elif idx in color_idx:
        return 1
    elif idx in brightness_idx:
        return 2
    elif idx in size_idx:
        return 3
    elif idx in q_spatial_rel_idx:
        return 4


def map_question_idx_to_group(idx, single_image, spatial_only):
    if (not single_image) and spatial_only:
        raise ValueError('Impossible spatial relation between separated objects.')
    if spatial_only:
        return map_question_idx_to_group_spatial_only(idx)
    else:
        return map_question_idx_to_group_all(idx, single_image)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def change_size(img, digit_size):
    """ We rescale the digit and we put it to the center.
    GIVE PRIORITY TO THIS TRANSFORMATION OVER ANY OTHER.
    :param img: image as numpy array
    :param digit_size: dimension of the digit
    """
    image = Image.fromarray(img)
    output_image = image.resize(size=(digit_size, digit_size))

    return np.array(output_image)


def apply_color_brightness(img, output_color=None, bright_level=1):
    """ We color the digit and fix the brightness for the objects.
     :param img: image as numpy array, 2D array
     :param output_color: string, output color
     :param bright_level: float, value in (0, 1]
     :returns output_image: 3D numpy array """
    if output_color is None:
        return img * bright_level
    output_image = np.zeros((3, img.shape[0], img.shape[1]))
    for id_ch_, ch_ in enumerate(color_dict[output_color]):
        output_image[id_ch_, :, :] = ch_ * img * bright_level
    return output_image


def transform(img, reshape, color, bright='light'):
    """ Transform the image based on the parameters we pass. In order
    Upscale
    Colors & Brightness
    :param img: input image
    :param reshape: string, image new size: small, medium, large
    :param color: digit color
    :param bright: string for brightness_dict
    """
    img_ = change_size(img, resize_dict[reshape])
    if np.max(img_) > 1:
        img_ = img_ / np.max(img_)

    img_ = apply_color_brightness(img_,
                                  output_color=color,
                                  bright_level=brightness_dict[bright])
    return img_


def generate_combinations(n_combinations_train=1,
                          n_combinations_test=5,
                          vocab=None):
    """ Here, we generate the combinations of training and test point.
    We base this on vocab.
    We care of keeping some combinations at test across all the experiments.
    Test set is shared across all the experiments which share the same question.
    Deterministic way of generating the data. We start by fixing the amount of
    objects that we will see at training.
    All the attributes must appear at least once.
    :param n_combinations_train: how many repetitions for all the attributes we have at train
    :param  n_combinations_test: how many repetitions for all the attributes we have at test
    :param vocab: dictionary containing all possible questions
    :return combinations_train: list of lists with all tuples at training
    :return combinations_test: list of lists with all tuples at test
    """
    if vocab is None:
        raise ValueError('You need to provide a vocabulary')
    combinations_train = []  # combinations for training
    combinations_test = []   # combinations for validation and test
    from_id_to_token = invert_dict(vocab["all_questions_token_to_idx"])

    category_ = []    # list of categories,
    color_ = []       # colors,
    brightness_ = []  # brightness,
    size_ = []        # and sizes from which to extract
    choice_list = [False for k_ in range(4)]  # cat, color, bright, size

    for val_ in ATTRIBUTES_DICT.values():
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

    # e.g., if None attributes is asked for "color",
    # we enter in the second condition
    if not choice_list[0]:
        category_ = [vocab["all_questions_token_to_idx"][str(0)]]
    if not choice_list[1]:
        color_ = [vocab["all_questions_token_to_idx"]["red"]]
    if not choice_list[2]:
        brightness_ = [vocab["all_questions_token_to_idx"]["light"]]
    if not choice_list[3]:
        size_ = [vocab["all_questions_token_to_idx"]["large"]]
    # we force the dataset to be red

    # dataset 15-19 random seed 123
    np.random.seed(123)  # in this way the tuples are always the same datasets 0-5
    # np.random.seed(456)  # for datasets 6-11
    # np.random.seed(789)  # for datasets 12-17
    # especially at test

    for k_ in range(n_combinations_test):

        generate = True  # no saved tuple
        # we force tuple that appeared at this round not to appear anymore
        # (1, pink, large, bright)
        # (2, red, large, bright)
        # (2, pink, large, bright) is not allowed
        # through dict_control
        dict_control = {k_: False for k_ in ATTRIBUTES_DICT.keys()}

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
        dict_control = {k_: False for k_ in ATTRIBUTES_DICT.keys()}

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

    return combinations_train, combinations_test


def map_relation_to_attribute(id_rel, spatial_only):
    if spatial_only:
        return -1
    else:
        if id_rel in range(21, 25):
            return 3
        if id_rel in range(25, 27):
            return 1
        if id_rel in range(29, 33):
            return 2
        if id_rel in range(27, 29):
            return 0
        if id_rel in range(33, 37):
            return -1
        else:
            raise ValueError("No match")


def map_attribute_id_to_attribute(idx):
    """Map attributes into its id."""
    if idx in category_idx:
        return 0
    elif idx in color_idx:
        return 1
    elif idx in brightness_idx:
        return 2
    elif idx in size_idx:
        return 3
    else:
        raise ValueError("The inserted index is wrong")


def generate_templates(combinations,
                       single_image,
                       spatial_only,
                       ):
    """ Given the relations, we extract combinations and generate the dataset."""
    pos_answ = {}
    neg_answ = {}

    if not single_image:
        relations = RELATIONS_DICT_EXP2
    elif spatial_only:
        relations = RELATIONS_SPATIAL_ONLY
    else:
        relations = RELATIONS

    for rkey, rval in relations.items():
        # print(rkey, rval)
        attribute_type = map_relation_to_attribute(rval, spatial_only)
        pos_answ_for_r = []
        neg_answ_for_r = []

        if rkey.startswith('same') or rkey.startswith('different'):
            # symmetrical
            for id_c_, c_ in enumerate(combinations):
                idxs_no_id_c_ = np.arange(len(combinations))
                idxs_no_id_c_ = np.delete(idxs_no_id_c_, id_c_)
                # code_attr_tp_lhs = ATTRIBUTES_DICT[c_[attribute_type]]

                for id_c2__ in idxs_no_id_c_:
                    c__ = combinations[id_c2__]
                    # code_attr_tp_rhs = ATTRIBUTES_DICT[c__[attribute_type]]

                    if rkey.startswith('same'):
                        if c_[attribute_type] == c__[attribute_type]:
                            pos_answ_for_r.append([c_, c__])
                        else:
                            neg_answ_for_r.append([c_, c__])

                    elif rkey.startswith('different'):
                        if c_[attribute_type] != c__[attribute_type]:
                            pos_answ_for_r.append([c_, c__])
                        else:
                            neg_answ_for_r.append([c_, c__])

        elif attribute_type == -1:
            # pick random tuples and generate positive and negative
            # we will control their relative positions during image generation
            for id_c_, c_ in enumerate(combinations):
                idxs_no_id_c_ = np.delete(np.arange(len(combinations)), id_c_)
                for id_c2__ in idxs_no_id_c_:
                    c__ = combinations[id_c2__]
                    pos_answ_for_r.append([c_, c__])
                    neg_answ_for_r.append([c_, c__])

        else:
            for id_c_, c_ in enumerate(combinations):
                code_attr_tp_lhs = ATTRIBUTES_DICT[c_[attribute_type]]
                idxs_no_id_c_ = np.arange(len(combinations))
                idxs_no_id_c_ = np.delete(idxs_no_id_c_, id_c_)

                for id_c2__ in idxs_no_id_c_:
                    c__ = combinations[id_c2__]
                    code_attr_tp_rhs = ATTRIBUTES_DICT[c__[attribute_type]]

                    if rkey == 'smaller':  # 'small': 18, 'medium': 19, 'large': 20
                        if code_attr_tp_lhs < code_attr_tp_rhs:
                            pos_answ_for_r.append([c_, c__])
                        elif code_attr_tp_lhs > code_attr_tp_rhs:
                            neg_answ_for_r.append([c_, c__])

                    elif rkey == 'larger':  # 'small': 18, 'medium': 19, 'large': 20
                        if code_attr_tp_lhs > code_attr_tp_rhs:
                            pos_answ_for_r.append([c_, c__])
                        elif code_attr_tp_lhs < code_attr_tp_rhs:
                            neg_answ_for_r.append([c_, c__])

                    elif rkey == 'brighter':  #  'dark': 15, 'half': 16, 'light': 17,
                        if code_attr_tp_lhs > code_attr_tp_rhs:
                            pos_answ_for_r.append([c_, c__])
                        elif code_attr_tp_lhs < code_attr_tp_rhs:
                            neg_answ_for_r.append([c_, c__])

                    elif rkey == 'darker':
                        if code_attr_tp_lhs < code_attr_tp_rhs:
                            pos_answ_for_r.append([c_, c__])
                        elif code_attr_tp_lhs > code_attr_tp_rhs:
                            neg_answ_for_r.append([c_, c__])

        pos_answ[rkey] = pos_answ_for_r
        neg_answ[rkey] = neg_answ_for_r

    return pos_answ, neg_answ


class Object(object):
    def __init__(self, pos=None, shape=None, digit_size=28, image_size=64):
        self.pos = pos
        self.shape = shape
        self.digit_size = digit_size
        self.image_size = image_size
        if self.pos is None:
            self.set_position()

    def set_position(self):
        posx = np.random.choice(np.arange(self.image_size - self.digit_size))
        posy = np.random.choice(np.arange(self.image_size - self.digit_size))
        self.pos = [posx, posy]

    def overlap(self, other):
        # min_dist = (self.rotated_size + other.rotated_size) // 2 + 1
        # digit_size is also min_dist
        return (abs(self.pos[0] - other.pos[0]) < (self.digit_size + other.digit_size) / 2 and
                abs(self.pos[1] - other.pos[1]) < (self.digit_size + other.digit_size) / 2)

    def relate(self, rel, other):
        shift = 0
        if rel == 'left_of':
            return self.pos[0] + shift < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0] + shift
        if rel == 'above':
            return self.pos[1] > other.pos[1] + shift
        if rel == 'below':
            return self.pos[1] + shift < other.pos[1]
        raise ValueError(rel)


class DataGenerator(object):
    def __init__(self,
                 data_path,
                 variety,
                 ch=3,
                 image_size=64,
                 single_image=True,
                 spatial_only=False,
                 gen_in_distr_test=False):
        """
        Dataset generation:
        :param data_path: path to the mnist dataset
        :param variety: min amount of time for which we see an attribute
        :param ch: number of channels
        :param image_size: size of the image (we assume it is square)
        :param single_image: experiment_4 or 2
        :param gen_in_distr_test: if True, we do not generate tuples, but we assign old from training
        """

        if (not single_image) and spatial_only:
            raise ValueError('Contradictory instruction')

        self.variety = variety
        self.ch = ch
        self.image_size = image_size
        self.single_image = single_image
        self.spatial_only = spatial_only
        self.gen_in_distr_test = gen_in_distr_test
        self.data_path = data_path
        self.loaded_mnist = False
        self.indices_per_category = None

        print(single_image)
        print(spatial_only)

        self.vocab = generate_vocab(single_image, spatial_only)

        if not gen_in_distr_test:
            self.train_combinations = None
            self.test_combinations = None

            comb_tr, comb_ts = generate_combinations(n_combinations_train=self.variety, vocab=self.vocab)
            pos_comb_tr, neg_comb_tr = generate_templates(comb_tr, single_image, spatial_only)
            pos_comb_ts, neg_comb_ts = generate_templates(comb_ts, single_image, spatial_only)

            self.pos_comb_tr = pos_comb_tr
            self.neg_comb_tr = neg_comb_tr
            self.pos_comb_ts = pos_comb_ts
            self.neg_comb_ts = neg_comb_ts

        print(self.vocab)
        self.vocab_q = self.vocab["question_token_to_idx"]

        # we need those combinations to be passed though

    def load_mnist(self, split):
        self.X = np.load(join(self.data_path, "x_%s.npy" % split))
        self.y = np.load(join(self.data_path, "y_%s.npy" % split))
        self.indices_per_category = [np.argwhere(self.y == k_).squeeze()
                                     for k_ in range(10)]

    def _generate_img(self, tuple_lhs, tuple_rhs, spatial=False, question=None, label=None):
        digit_size_lhs = resize_dict[tuple_lhs[-1]]
        digit_size_rhs = resize_dict[tuple_rhs[-1]]

        id_obj_lhs = np.random.choice(self.indices_per_category[self.vocab_q[tuple_lhs[0]]])
        id_obj_rhs = np.random.choice(self.indices_per_category[self.vocab_q[tuple_rhs[0]]])

        flag_overlap = True if self.single_image else False

        while flag_overlap:
            o_lhs = Object(shape=tuple_lhs[0], digit_size=digit_size_lhs, image_size=self.image_size)
            o_rhs = Object(shape=tuple_rhs[0], digit_size=digit_size_rhs, image_size=self.image_size)
            flag_overlap = o_lhs.overlap(o_rhs)

        if not self.single_image:
            pos_xy_lhs = self.image_size // 2 - digit_size_lhs // 2
            pos_xy_rhs = self.image_size // 2 - digit_size_rhs // 2
            o_lhs = Object(shape=tuple_lhs[0], digit_size=digit_size_lhs, image_size=self.image_size,
                           pos=[pos_xy_lhs, pos_xy_lhs])
            o_rhs = Object(shape=tuple_rhs[0], digit_size=digit_size_rhs, image_size=self.image_size,
                           pos=[pos_xy_rhs, pos_xy_rhs])

        if spatial:
            while label != o_lhs.relate(rel=question, other=o_rhs) or flag_overlap:
                o_lhs = Object(shape=tuple_lhs[0], digit_size=digit_size_lhs, image_size=self.image_size)
                o_rhs = Object(shape=tuple_rhs[0], digit_size=digit_size_rhs, image_size=self.image_size)
                flag_overlap = o_lhs.overlap(o_rhs)

        img = np.zeros((self.ch, self.image_size, self.image_size))
        if not self.single_image:
            img_list = []

        img_lhs = transform(self.X[id_obj_lhs] / 255,
                            reshape=tuple_lhs[3],
                            color=tuple_lhs[1],
                            bright=tuple_lhs[2])
        dig_size = img_lhs.shape[1]
        yy, xx = np.meshgrid(np.arange(o_lhs.pos[0], o_lhs.pos[0] + dig_size),
                             np.arange(o_lhs.pos[1], o_lhs.pos[1] + dig_size))
        img[:, xx, yy] = img_lhs
        if not self.single_image:
            img_list.append(img)

        img_rhs = transform(self.X[id_obj_rhs] / 255,
                            reshape=tuple_rhs[3],
                            color=tuple_rhs[1],
                            bright=tuple_rhs[2])
        dig_size = img_rhs.shape[1]
        yy, xx = np.meshgrid(np.arange(o_rhs.pos[0], o_rhs.pos[0] + dig_size),
                             np.arange(o_rhs.pos[1], o_rhs.pos[1] + dig_size))
        if self.single_image:
            img[:, xx, yy] = img_rhs
        else:
            img = np.zeros((self.ch, self.image_size, self.image_size))
            img[:, xx, yy] = img_rhs
            img_list.append(img)

            img = np.zeros((self.ch, self.image_size, self.image_size, 2))
            img[:, :, :, 0] = img_list[0]
            img[:, :, :, 1] = img_list[1]

        return img

    def _generate_attribute_pair(self, comb, question, label):
        """:param comb: a couple of objects, e.g.
        ['3', 'yellow', 'dark', 'large'], ['0', 'blue', 'half', 'large']
        # control: call is not from the same attribute
            # given ['3', 'yellow', 'dark', 'large'], ['0', 'blue', 'half', 'large']
            # [large, left_of, large]
            # or a shared attribute

            # ['6', 'yellow', 'dark', 'large'], ['6', 'blue', 'half', 'large']

            # [6, left_of, large]
        """
        c_lhs = comb[0]
        c_rhs = comb[1]
        diff_attribute = np.argwhere(np.array([lhs_ != rhs_
                                               for lhs_, rhs_ in zip(c_lhs, c_rhs)])).squeeze()

        if diff_attribute.size == 1:
            lhs__ = c_lhs[diff_attribute]
            rhs__ = c_rhs[diff_attribute]
        else:
            lhs__ = c_lhs[np.random.choice(diff_attribute)]
            rhs__ = c_rhs[np.random.choice(diff_attribute)]

        spatial = question in ['right_of', 'left_of', 'above', 'below']

        return lhs__, rhs__, spatial, question, label

    def generate_data_matrix(self,
                             n=[210000, 42000, 42000],
                             savepath=None,
                             h5_file=False):
        split_list = ["train", "valid", "test"]
        if isinstance(n, int):
            n = [n, n, n]

        if self.gen_in_distr_test:
            n = n[1:]
            split_list = split_list[1:]
            self.pos_comb_tr = json.load(open(join(savepath, 'pos_train.json'), 'rb'))
            self.neg_comb_tr = json.load(open(join(savepath, 'neg_train.json'), 'rb'))

        else:
            with open(join(savepath, 'pos_train.json'), 'w') as outfile:
                json.dump(self.pos_comb_tr, outfile)
            with open(join(savepath, 'pos_test.json'), 'w') as outfile:
                json.dump(self.pos_comb_ts, outfile)
            with open(join(savepath, 'neg_train.json'), 'w') as outfile:
                json.dump(self.neg_comb_tr, outfile)
            with open(join(savepath, 'neg_test.json'), 'w') as outfile:
                json.dump(self.neg_comb_ts, outfile)

        for split_, n_ in zip(split_list[::-1], n[::-1]):  # for each split
            if self.gen_in_distr_test or split_ == 'train':
                p_combinations, n_combinations = self.pos_comb_tr, self.neg_comb_tr
            else:
                p_combinations, n_combinations = self.pos_comb_ts, self.neg_comb_ts

            # print(p_combinations)
            n_questions = len(p_combinations.keys())
            examples_per_q = (n_ // 2) // n_questions  # divided into positive and negative

            if h5_file:
                f = h5py.File(join(savepath, "feats_%s.hdf5" % split_), 'w')
                if self.single_image:
                    dset = f.create_dataset('features',
                                            shape=(2*examples_per_q, self.ch, self.image_size, self.image_size),
                                            maxshape=(n_, self.ch, self.image_size, self.image_size),
                                            dtype=np.float64,
                                            chunks=True)
                else:
                    dset = f.create_dataset('features',
                                            # shape=(examples_per_q, self.ch, self.image_size, self.image_size, 2),
                                            maxshape=(n_, 2, self.ch, self.image_size, self.image_size),
                                            shape=(2*examples_per_q, 2, self.ch, self.image_size, self.image_size),
                                            dtype=np.float64,
                                            chunks=True)
            else:
                if self.single_image:
                    x_out = np.zeros((n_, self.ch, self.image_size, self.image_size))
                else:
                    x_out = np.zeros((n_, self.ch, self.image_size, self.image_size, 2))
            q_out = np.zeros((n_, 3), dtype=int)
            y_out = np.zeros(n_, dtype=int)
            a_out = []
            self.load_mnist(split_)

            np.random.seed(42)
            count = 0
            # TODO change
            # n_questions = 1
            for id_q__, q_ in enumerate(list(p_combinations.keys())[:]):  # TODO! change this index

                if h5_file:  # for each question we have a new matrix
                    index_img = 0
                    if self.single_image:
                        x_out = np.zeros((2*examples_per_q, self.ch, self.image_size, self.image_size))
                    else:
                        x_out = np.zeros((2*examples_per_q, self.ch, self.image_size, self.image_size, 2))
                else:
                    index_img = count
                print("\nQuestion: ", q_)
                class_comb_per_q_pos = p_combinations[q_]
                class_comb_per_q_neg = n_combinations[q_]

                n_class_combs_pos = len(class_comb_per_q_pos)  # .shape[0]
                n_class_combs_neg = len(class_comb_per_q_neg)  # .shape[0]

                examples_per_el_pos = (examples_per_q // n_class_combs_pos)
                examples_per_el_neg = (examples_per_q // n_class_combs_neg)

                print("# combinations available")
                print(n_class_combs_pos, n_class_combs_neg)

                for i, comb in enumerate(class_comb_per_q_pos):
                    for l_ in range(examples_per_el_pos):
                        label = 1
                        c_lhs, c_rhs, spatial, _, _ = self._generate_attribute_pair(comb,
                                                                                    question=q_,
                                                                                    label=label)
                        # print(comb[0], q_, comb[1], spatial)
                        x_out[index_img, :, :, :] = self._generate_img(tuple_lhs=comb[0],
                                                                       tuple_rhs=comb[1],
                                                                       spatial=spatial,
                                                                       question=q_,
                                                                       label=label)
                        q_out[count] = np.array([self.vocab_q[c_lhs],
                                                 self.vocab_q[q_],
                                                 self.vocab_q[c_rhs]])
                        y_out[count] = label
                        count += 1
                        index_img += 1
                        a_out.append([comb[0], q_, comb[1]])

                for i, comb in enumerate(class_comb_per_q_neg):
                    for l_ in range(examples_per_el_neg):
                        label = 0
                        c_lhs, c_rhs, spatial, _, _ = self._generate_attribute_pair(comb,
                                                                                    question=q_,
                                                                                    label=label)
                        x_out[index_img, :, :, :] = self._generate_img(tuple_lhs=comb[0],
                                                                       tuple_rhs=comb[1],
                                                                       spatial=spatial,
                                                                       question=q_,
                                                                       label=label)
                        q_out[count] = np.array([self.vocab_q[c_lhs],
                                                 self.vocab_q[q_],
                                                 self.vocab_q[c_rhs]])
                        y_out[count] = label
                        count += 1
                        index_img += 1
                        a_out.append([comb[0], q_, comb[1]])

                while count < 2 * examples_per_q * (id_q__+1):
                    # rnd_q_idx = np.random.choice(len(p_combinations))  # questions
                    # q_ = p_combinations.keys()[rnd_q_idx]
                    label = count % 2
                    if label:
                        combinations_per_q = p_combinations[q_]
                    else:
                        combinations_per_q = n_combinations[q_]
                    n_el = len(combinations_per_q)
                    id__rdn = np.random.choice(n_el) if n_el > 1 else 0
                    comb = combinations_per_q[id__rdn]
                    c_lhs, c_rhs, spatial, _, _ = self._generate_attribute_pair(comb,
                                                                                question=q_,
                                                                                label=label)
                    x_out[index_img, :, :, :] = self._generate_img(tuple_lhs=comb[0],
                                                                   tuple_rhs=comb[1],
                                                                   spatial=spatial,
                                                                   question=q_,
                                                                   label=label)
                    q_out[count] = np.array([self.vocab_q[c_lhs],
                                             self.vocab_q[q_],
                                             self.vocab_q[c_rhs]])
                    y_out[count] = label
                    count += 1
                    index_img += 1
                    a_out.append([comb[0], q_, comb[1]])

                if h5_file:
                    if id_q__ > 0:
                        dset.resize(count, axis=0)
                    if self.single_image:
                        dset[id_q__*2*examples_per_q:] = x_out
                    else:
                        dset[id_q__*2*examples_per_q:] = x_out.transpose(0, 4, 1, 2, 3)
                    # save in the hdf5 file
                    # 2 * examples_per_q at the end of each question

            missing_examples = n_ - count

            if h5_file:
                index_img = 0
                if self.single_image:
                    x_out = np.zeros((missing_examples, self.ch, self.image_size, self.image_size))
                else:
                    x_out = np.zeros((missing_examples, self.ch, self.image_size, self.image_size, 2))
            else:
                index_img = count

            while count < n_:

                rnd_q_idx = np.random.choice(len(p_combinations))  # questions
                q_ = list(p_combinations.keys())[rnd_q_idx]
                label = count % 2
                if label:
                    combinations_per_q = p_combinations[q_]
                else:
                    combinations_per_q = n_combinations[q_]
                n_el = len(combinations_per_q)
                id__rdn = np.random.choice(n_el) if n_el > 1 else 0
                comb = combinations_per_q[id__rdn]
                c_lhs, c_rhs, spatial, _, _ = self._generate_attribute_pair(comb,
                                                                            question=q_,
                                                                            label=label)
                x_out[index_img, :, :, :] = self._generate_img(tuple_lhs=comb[0],
                                                               tuple_rhs=comb[1],
                                                               spatial=spatial,
                                                               question=q_,
                                                               label=label)
                q_out[count] = np.array([self.vocab_q[c_lhs],
                                         self.vocab_q[q_],
                                         self.vocab_q[c_rhs]])
                y_out[count] = label
                a_out.append([comb[0], q_, comb[1]])

                count += 1
                index_img += 1
                # print(count)

            if h5_file:
                dset.resize(n_, axis=0)
                if self.single_image:
                    dset[n_ - missing_examples:] = x_out
                else:
                    dset[n_-missing_examples:] = x_out.transpose(0, 4, 1, 2, 3)

            rnd_idx = np.arange(count)
            # np.random.shuffle(rnd_idx) # TODO: remove comment

            if self.gen_in_distr_test:
                split_ = 'in_distr_' + split_

            if self.single_image:
                # if h5_file:
                #     with h5py.File(join(savepath, "feats_%s.hdf5" % split_), "w") as f:
                #          #  (4,4), dtype='i', data=A)
                #         f.create_dataset("features", x_out[rnd_idx].shape, data=x_out[rnd_idx])
                #         f.close()
                if not h5_file:
                    np.save(join(savepath, "feats_%s.npy" % split_), x_out[rnd_idx])
            else:
                # if h5_file:
                #     with h5py.File(join(savepath, "feats_%s.hdf5" % split_), "w") as f:
                #         shape_ = x_out[rnd_idx].transpose(0, 4, 1, 2, 3).shape
                #         f.create_dataset("features", shape_, data=x_out[rnd_idx].transpose(0, 4, 1, 2, 3))
                #         f.close()
                if not h5_file:
                    np.save(join(savepath, "feats_%s.npy" % split_), x_out[rnd_idx].transpose(0, 4, 1, 2, 3))

            np.save(join(savepath, "answers_%s.npy" % split_), y_out[rnd_idx])
            np.save(join(savepath, "questions_%s.npy" % split_), q_out[rnd_idx])

            with open(join(savepath, 'attributes_%s.pkl' % split_), 'wb') as f:
                pickle.dump(a_out, f)

            # print("COUNT %s: %i" % (split_, count))

        if not self.gen_in_distr_test:
            with open(join(savepath, 'vocab.json'), 'w') as outfile:
                json.dump(self.vocab, outfile)

        return
