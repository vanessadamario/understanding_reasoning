import numpy as np
import json
from os.path import join, dirname
from runs.data_attribute_random import category_idx, color_idx, brightness_idx, size_idx
from runs.data_attribute_random import invert_dict
from runs.mnist_attribute_generation import transform


def generate_activation_dataset(data_path, experiment_case, n_el=100):

    vocab = json.load(open(join(data_path, 'vocab.json'), 'rb'))

    x = np.load(join(dirname(dirname(data_path)), 'MNIST_splits', 'x_test.npy'))
    y = np.load(join(dirname(dirname(data_path)), 'MNIST_splits', 'y_test.npy'))

    dims = np.load(join(data_path, 'feats_test.npy')).shape
    dims = dims[1:]

    flag_query = 'query_' if experiment_case == 0 else ''
    if experiment_case == 0:
        n_combinations = len(category_idx) * len(color_idx) * len(brightness_idx) * len(size_idx)
        x_new = np.zeros((n_el * n_combinations, ) + dims)
        q_new = np.zeros((n_el * n_combinations, ) + (4,), dtype=int)
        idx_per_class = []
        for i in category_idx:  # the idx matches the category
            idx_per_class.append(np.argwhere(y == i).squeeze())

        combinations = []
        count = 0

        inverted_ = invert_dict(vocab['question_token_to_idx'])
        for cat_ in category_idx:
            for col_ in color_idx:
                for bri_ in brightness_idx:
                    for siz_ in size_idx:
                        rnd_idx = np.random.choice(idx_per_class[cat_], size=n_el)
                        for id_image in rnd_idx:
                            x_new[count] = transform(x[id_image] / 255,
                                                     reshape=inverted_[siz_],
                                                     color=inverted_[col_],
                                                     bright=inverted_[bri_],
                                                     contour=False)
                            combinations.append([inverted_[cat_],
                                                 inverted_[col_],
                                                 inverted_[bri_],
                                                 inverted_[siz_]])
                            q_new[count] = np.array([0, 1, 2, 3])
                            count += 1
    else:
        raise NotImplementedError('Missing case')

    np.save(join(data_path, 'feats_%sactivation.npy' % flag_query), x_new)
    np.save(join(data_path, 'questions_%sactivation.npy' % flag_query), q_new)
    np.save(join(data_path, 'attributes_%sactivation.npy' % flag_query), np.array(combinations))


def sample_selection(data_path,
                     experiment_case):
    flag_query = 'query_' if experiment_case == 0 else ''
    distrib_type = ['in_distr_', '']
    for flag_d_ in distrib_type:
        # do we need seen and unseen tuples
        # x = np.load(join(data_path, 'feats_%stest.npy' % flag_d_))
        # q = np.load(join(data_path, 'questions_%stest.npy' % (flag_query + flag_d_)))
        # y = np.load(join(data_path, 'answers_%stest.npy' % (flag_query + flag_d_)))
        a = np.load(join(data_path, 'attributes_%stest.npy' % flag_d_))

        if experiment_case == 0:  # query case
            unique_a = np.unique(a, axis=0)  # all tuples
            mask = []
            for id_unique, unique_ in enumerate(unique_a):  # for each tuple (combination)
                tmp = np.array([np.all(unique_ == a_) for a_ in a])
                mask_tmp = np.argwhere(tmp).squeeze()  # where in the file
                mask.append(np.random.choice(mask_tmp, size=100))

            mask = np.array(mask)
            np.save(join(data_path, '%sindexes_activations.npy' % (flag_query + flag_d_)),
                    mask)

        else:
            raise NotImplementedError('VQA is missing')