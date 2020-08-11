from PIL import Image
import os
import io
import json
import torch
import numpy as np
import PIL
from vr.utils import load_execution_engine, load_program_generator
from vr.models.shnmn import _shnmn_func
from torch.autograd import Variable
from os.path import join
import PIL.Image
import h5py


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_name = {2: dict(), 5: dict()}
lhs_lst = [1, 2, 4, 8, 18, 35]
n_per_lhs_lst = [30000, 15000, 7500, 3750, 2000, 1000]
for lhs_, n_ in zip(lhs_lst, n_per_lhs_lst):
    data_name[2][lhs_] = 'sqoop-no_crowding-variety_%i-repeats_%i' % (lhs_, n_)
    data_name[5][lhs_] = 'sqoop-variety_%i-repeats_%i' % (lhs_, n_)


def load_data_(data_path, file_str='test_in_sample_2'):
    """ Load dataset and return a list of features, questions, and answers
    over which we can iterate
    :param data_path: str
    :param file_str: str starting name of the file
    :return features: list of numpy arrays
    :return questions: list of array, each with three components
    :return answers: list of array, with true or false value
    """
    feats_h5 = h5py.File(join(data_path, '%s_features.h5' % file_str), 'r')
    questions_h5 = h5py.File(join(data_path, '%s_questions.h5' % file_str), 'r')
    return list(feats_h5['features']), list(questions_h5['questions']), list(questions_h5['answers'])


def feats_to_image(feats):
    """ Convert the features into an image.
    :param feats: numpy array of features
    :returns image: the image with (3, dim_x, dim_y)
    """
    return np.array(PIL.Image.open(io.BytesIO(feats))).transpose(2, 0, 1) / 255.0


def load_shnmn_model_eval(model_path):
    """ Load the modular networks
    :param model_path: str, path to the model
    """
    model, kwargs = load_execution_engine(model_path)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    args_model = json.load(open('%s.pt.json' % (model_path.split('.pt.best')[0]), 'rb'))

    return model, args_model['args']


def compute_top_n_activations(model_path, output_path, feats, questions, top_n=9):
    """
    Compute the top-nine activations - returns images indexes and max values
    for the normalization factor.
    :param model_path: str, model path as loaded by load_shnmn_model_eval
    :param output_path: str, output path to save the files
    :param feats: list of np.array containing image features
    :param questions: list of np.array containing the questions
    :param top_n: int, number for which we evaluate the top activations
    :return max_activations_idx_top_n: image indexes for the top_n activations for the three modules
    :return: lowest_max_activation_top_n: lowest top_n max activations
    """
    model, args = load_shnmn_model_eval(model_path)
    num_modules = model.num_modules
    module_dim = args['module_dim']
    chs, dim_x, dim_y = args['feature_dim']

    max_activations_idx_top_n = np.zeros((top_n, num_modules, module_dim), dtype=int)
    lowest_max_activation_top_n = np.zeros((top_n, num_modules, module_dim))

    for idx_, (feats_, quest_) in enumerate(zip(feats, questions)):
        feats_ = feats_to_image(feats_)
        question_torch = torch.tensor(quest_)
        feats_torch = torch.tensor((feats_.reshape(1, chs, dim_x, dim_y)).astype('float32'))

        question_var = Variable(question_torch.to(device))
        feats_var = Variable(feats_torch.to(device))

        question_embed = model.question_embeddings(question_var)
        stem_image = model.stem(feats_var)

        res = _shnmn_func(question=question_embed,
                          img=stem_image.unsqueeze(1),
                          num_modules=model.num_modules,
                          alpha=model.alpha,
                          tau_0=Variable(model.tau_0),
                          tau_1=Variable(model.tau_1),
                          func=model.func)
        # typical dimensions going in: 1, 5, 64, 16, 16
        # batch_size, modules, dimensions_of_modules, dim_x input feat map, dim_y input feat map,
        maps = res.cpu().detach().numpy()[0]  # reduce the dimension 0
        tmp_max = np.max(maps[2:], axis=(2, 3))  # consider the X, R, Y modules
        # max across activation

        if idx_ < top_n:
            max_activations_idx_top_n[idx_] = idx_
            lowest_max_activation_top_n[idx_] = tmp_max

            if idx_ == top_n-1:
                max_activations_idx_top_n = np.argsort(lowest_max_activation_top_n, axis=0)
                lowest_max_activation_top_n = np.sort(lowest_max_activation_top_n, axis=0)
        else:
            bm = tmp_max > lowest_max_activation_top_n[0]  #  this is now 3 times 64
            # we are substituting in the smallest value
            lowest_max_activation_top_n[0, bm] = tmp_max[bm]
            max_activations_idx_top_n[0, bm] = idx_
            idx_sorted = np.argsort(lowest_max_activation_top_n, axis=0)
            lowest_max_activation_top_n = np.sort(lowest_max_activation_top_n, axis=0)

            for i__ in range(num_modules):
                for j__ in range(module_dim):
                    max_activations_idx_top_n[:, i__, j__] = max_activations_idx_top_n[idx_sorted[:, i__, j__], i__, j__]

    normalization_constants = lowest_max_activation_top_n[-1, :, :]

    np.save('%s_norm_const.npy' % output_path, normalization_constants)
    np.save('%s_idx_images.npy' % output_path, max_activations_idx_top_n)

    return max_activations_idx_top_n, lowest_max_activation_top_n


def compute_activation_maps(model_path, output_path, feats, quest, normalization=True):
    """
    We compute the activation maps for the top images that activated the most.
    :param model_path: str, path to the model
    :param output_path: str, output path to save the files
    :param feats: list of np.array, all the elements used for compute_top_n_activations
    :param quest: list of np.array, all the questions used as aboce
    :param normalization: np.array, normalization constant
    :return: activation_maps, np.array containing the activation maps
    """
    model, args = load_shnmn_model_eval(model_path)
    num_modules = model.num_modules
    module_dim = args['module_dim']
    chs, dim_x, dim_y = args['feature_dim']

    if normalization:
        norm_constant = np.load('%s_norm_const.npy' % (model_path.split('.pt.best')[0]))
    else:
        norm_constant = np.ones((num_modules, module_dim))
    max_activations_id_img = np.load('%s_idx_images.npy' % (model_path.split('.pt.best')[0]))

    top_n = max_activations_id_img.shape[0]

    for i in range(num_modules):  # for each module
        for j in range(module_dim):  # for each dimension
            for count_, id_img_ in enumerate(max_activations_id_img[:, i, j]):
                feats_ = feats_to_image(feats[id_img_])
                feats_ = feats_.reshape(1, chs, dim_x, dim_y).astype('float32')

                question_torch = torch.tensor(quest[id_img_])
                feats_torch = torch.tensor(feats_)

                question_var = Variable(question_torch.to(device))
                feats_var = Variable(feats_torch.to(device))

                question_embed = model.question_embeddings(question_var)
                stem_image = model.stem(feats_var)

                res = _shnmn_func(question=question_embed,
                                  img=stem_image.unsqueeze(1),
                                  num_modules=model.num_modules,
                                  alpha=model.alpha,
                                  tau_0=Variable(model.tau_0),
                                  tau_1=Variable(model.tau_1),
                                  func=model.func)
                maps = res.cpu().detach().numpy()[0]

                if i == 0 and j == 0:
                    dim_act = maps.shape[-1]
                    activation_maps = np.zeros((num_modules, module_dim, top_n, dim_act, dim_act))
                activation_maps[i, j, count_, :, :] = maps[i + 2, j, :, :] / norm_constant[i, j]

    np.save('%s_activation_maps.npy' % output_path, activation_maps)

    return


def main():
    root_path = './../..'

    # TODO: change the following 3 def based on the dataset and the models
    lhs = 18
    module_type = 'find'  # find or residual
    model_type = 'tree'  # tree, chain, chain_w_shortcuts, chain_w_shortcuts_flipped
    n_objects = 2

    path_to_dataset = join(root_path,
                           'datasets',
                           data_name[n_objects][lhs])
    folder_models = join(root_path,
                         'systematic-generalization-sqoop_2objects/results',
                         '%s_models_2objs' % model_type,
                         module_type,
                         'lhs%i' % lhs)

    lst_slurm_id = [f_ for f_ in os.listdir(folder_models) if f_.endswith('.pt.best')]

    for slurm_id in lst_slurm_id:
        print(slurm_id)
        feats_lst, quest_lst, _ = load_data_(path_to_dataset, file_str='test_in_sample')
        path_to_model = join(folder_models, slurm_id)
        output_path = join(path_to_model.split('.pt.best')[0])

        compute_top_n_activations(path_to_model, output_path, feats_lst, quest_lst)
        compute_activation_maps(path_to_model, output_path, feats_lst, quest_lst)


if __name__ == '__main__':
    main()