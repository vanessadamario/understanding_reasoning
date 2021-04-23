import os
import numpy as np
import pandas as pd
import torch
import collections
from os.path import join, dirname
from torch.autograd import Variable

from runs.data_attribute_random import category_idx, color_idx, brightness_idx, size_idx
from runs.dict_architectures_hyper import dct_hyper_method
from runs.utils import load_vocab
from runs.data_loader import DataTorch, DataLoader, _dataset_to_tensor
from runs.experiments import get_experiment
from runs.data_attribute_random import category_idx, color_idx, brightness_idx, size_idx


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_activations(architectures,
                        dataset,
                        path,
                        experiment_case=0,
                        exact_activations=False,
                        new_output_path=False,
                        new_data_path=False):
    """This is the function we call in the main.py to
    compute activations
    :param architectures: str, e.g., sep_res
    :param path: networks and train.json path
    :param experiment_case: 0 if query, 1 is vqa
    :param exact_activations: bool, False for vqa
    :param new_output_path: bool --
    :param new_data_path: bool -- these flags are useful when the path in the train.json
    does not coincide with the location of the experiments
    """
    json_file = join(path, "train.json")
    print('path')
    print(path)
    df = pd.read_json(json_file).T

    # given the architecture and the dataset
    # search for the best experiment
    dict_search = best_network(architectures, dataset, experiment_case)
    indexes = search_experiments(df, dict_search)  # return corresponding experiments
    if len(indexes) == 0:
        raise ValueError('No match')
    print(indexes)
    path_list = [join(path, 'train_%i' % k) for k in indexes] if new_output_path else None

    idx, ts = find_best_performance(df, indexes,
                                    provide_path=new_output_path,
                                    path_list=path_list)
    print('best index')
    print(idx)
    opt = get_experiment(path, idx)
    if new_output_path:
        opt.output_path = join(path, 'train_%i' % idx)
    if new_data_path:
        opt.dataset.dataset_id_path = join(os.path.dirname(path),
                                           'data_generation/datasets',
                                           opt.dataset.dataset_id)

    # generate new activation folder
    activation_path = join(path, 'activations', dataset)
    os.makedirs(activation_path, exist_ok=True)

    activation_path = join(activation_path, architectures)
    os.makedirs(activation_path, exist_ok=True)

    np.save(join(activation_path, 'index.npy'), idx)  # save index of the best model

    # we call the activation class and compute the activations
    ee = ActivationClass(opt,
                         activation_path=activation_path,
                         approx=not exact_activations,
                         in_distribution=False,
                         threshold=False,
                         )
    ee.compute(stem=True, module=True, classifier=True)


def best_network(architecture_name, dataset_name, experiment_case=0):
    """ We pass the arguments to create a searching dictionary
     :param architecture_name: name of the architectures
        one key among those in dict_architectures_hyper
     :param dataset_name: name of the dataset
     :param experiment_case: if 1 vqa, otherwise query
     """
    dict_search = {}
    dict_search['hyper_method'] = dct_hyper_method[architecture_name].copy()
    data_dict = {'dataset_id': dataset_name,
                 'experiment_case': experiment_case}
    dict_search['dataset'] = data_dict

    return dict_search


def search_experiments(df, common_hypers):
    """ Find experiments that share identical hyper-parameters
    :param df: pd.DataFrame
    :param common_hypers: dictionary of hyper-parameters
    :returns: indexes of df
    """
    print(common_hypers)
    n_exps = df.shape[0]
    same_exp_array = np.ones(n_exps, dtype=int)
    for id_ in range(n_exps):
        for k_, v_ in common_hypers.items():
            if type(v_) is dict:
                try:
                    same_exp = np.prod(np.array([df.iloc[id_][k_][k__] == v__
                                                 for k__, v__ in v_.items()]))
                except:
                    same_exp = False
            else:
                try:
                    same_exp = v_ == df.iloc[id_][k_]
                except:
                    same_exp = False
            if not same_exp:
                same_exp_array[id_] = same_exp
                break
    same_exp_in_df = np.squeeze(np.argwhere(same_exp_array))
    indexes = np.array([df.iloc[k_]['id'] for k_ in (same_exp_in_df)])
    return indexes


def find_best_performance(df,
                          indexes,
                          seen_valid=True,
                          seen_test=False,
                          provide_path=False,
                          path_list=None):
    """
        We need to run the run_test function in the main, before proceeding.

    We evaluate the best validation accuracy, as the training is completed, and specify if the cross
    validation is done with respect to the seen distribution or unseen
    :param df: pandas dataframe containing the specifics for experiments
    :param indexes: list of indexes
    :param seen_valid: if True, we consider the validation with same distribution as the training
    to perform cross validation, otherwise unseen distribution
    :param seen_test: if True, we return the test accuracy on seen distribution otherwise ood
    :param provide_path: bool, if true we want to use a path different from the df
    :param path_list: if provide path, we pass a list of paths for the df experiments with i_ indexes
    We return the index for the best model.
    """
    if not provide_path:
        # path_list = [df.iloc[k_]["output_path"] for k_ in indexes]
        path_list = df.loc[indexes, 'output_path']
    valid_list = []
    filename_valid = 'seen_valid_accuracy.npy' if seen_valid else 'valid_accuracy.npy'
    filename_test = 'seen_accuracy.npy' if seen_test else 'test_accuracy.npy'

    # print(filename_valid)
    try:  # if not all the experiments are available we return (-1,-1)
        for path_ in (path_list):
            print(path_)
            valid_list.append(np.load(join(path_, filename_valid)))
    except:
        return -1, -1

    valid_np = np.array(valid_list)

    if valid_np.ndim == 2:  # query case
        valid_list = np.mean(valid_np, axis=1)
    print(np.argmax(valid_list))
    best_cv = indexes[np.argmax(valid_list)]
    # print(best_cv-offset)
    if provide_path:
        path_best_cv = join(dirname(path_list[0]), 'train_%i' % best_cv)
        print(path_best_cv)
    else:
        path_best_cv = df.loc[best_cv]['output_path']
    test_accuracy = np.load(join(path_best_cv, filename_test))

    return indexes[np.argmax(valid_list)], test_accuracy


class DataLoaderActivations(DataLoader):
    """ Data loader for evaluation of the activations values.
    Do not modify the batch size, which matches with the number
    of example for a specific attribute combinations when the
    we do not use the approximation """
    def __init__(self, all_feats, all_questions, all_labels, batch_size=100):
        self.batch_size = batch_size
        all_questions = _dataset_to_tensor(all_questions)
        all_labels = _dataset_to_tensor(all_labels)
        # all_feats = np.load(all_feats)
        self.dataset = DataTorch(all_feats=all_feats,
                                 all_questions=all_questions,
                                 all_labels=all_labels)
        super(DataLoaderActivations, self).__init__(self.dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)


def update_values(existing_aggregate, new_value):
    """
    Welford's online algorithm to compute mean and standard deviation
    For a new value newValue, compute the new count, new mean, the new M2.
    mean accumulates the mean of the entire dataset
    M2 aggregates the squared distance from the mean
    count aggregates the number of samples seen so far
    :param existing_aggregate: tuple of count mean and sum of square
    :param new_value: new element to include in the statistics
    :return count, mean, M2: updated aggregate
    """
    (count, mean, M2) = existing_aggregate
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return count, mean, M2


def finalize_values(existing_aggregate):
    """Compute the mean, variance and sample variance given an aggregate
    """
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return mean, variance, sampleVariance


class ActivationClass(object):
    """
    In this class we compute the activation values
    at every layer of the network
    """
    def __init__(self,
                 opt,
                 activation_path,
                 model_path=None,
                 approx=False,
                 in_distribution=False,
                 threshold=False):
        """
        :param opt: object containing details about the experiment
        :param activation_path: where to save the activation
        :param model_path: model's path in case different from the opt
        :param approx: True for all the vqa case
        :param in_distribution: activations for objects in or out of distribution
        :param threshold: bool, False -- set to true for method with threshold and max
        """
        self.opt = opt
        self.activation_path = activation_path
        self.approx = approx
        self.in_distribution = in_distribution
        self.threshold = threshold

        # we recover the architecture and load the model
        vocab = load_vocab(join(opt.dataset.dataset_id_path, "vocab.json"))
        kkwargs_exec_engine_ = opt.hyper_method.__dict__
        kkwargs_exec_engine_["vocab"] = vocab
        kkwargs_exec_engine_["load_model"] = True
        if model_path is None:
            kkwargs_exec_engine_["output_path"] = opt.output_path
        else:
            kkwargs_exec_engine_["output_path"] = model_path

        if opt.dataset.experiment_case == 1:
            from runs.train_loop import get_execution_engine
        elif opt.dataset.experiment_case == 0:
            from runs.train_query_loop import get_execution_engine
        else:
            raise ValueError('experiment case not accepted')

        ee, _ = get_execution_engine(**kkwargs_exec_engine_)
        print(ee)
        ee.eval()  # we set the model into the evaluation mode
        self.ee = ee

        size_hypercube = [len(category_idx),
                          len(color_idx),
                          len(brightness_idx),
                          len(size_idx)]
        self.size_hypercube = size_hypercube
        self.vocab_q = vocab['question_token_to_idx']

    def map_into_combination(self, attribute):
        """ We pass the vocab and the attribute array
        """
        if attribute.size == 5:
            attribute = attribute[:-1]
        mapped_idx = [self.vocab_q[a_] for a_ in attribute]
        reset_idx = [category_idx[0], color_idx[0], brightness_idx[0], size_idx[0]]
        mapped_comb = [mapped_ - reset_ for mapped_, reset_ in zip(mapped_idx, reset_idx)]
        return mapped_comb

    def load_data(self):
        """ Depending if we use approximations or not
        we might load
            - approximation: test data (typically out of distribution)
            - exact: dataset for all possible attributes
        """
        flag_q = 'query_' if self.opt.dataset.experiment_case == 0 else ''
        if self.approx:
            flag_d = 'in_distr_' if self.in_distribution else ''
            feats = np.load(join(self.opt.dataset.dataset_id_path,
                                 'feats_' + flag_d + 'test.npy'))
            questions = np.load(join(self.opt.dataset.dataset_id_path,
                                     'questions_' + flag_q + flag_d + 'test.npy'))
            answers = np.zeros(questions.shape)
            attributes = np.load(join(self.opt.dataset.dataset_id_path,
                                      'attributes_' + flag_d + 'test.npy'))
            return flag_q, flag_d, questions, feats, answers, attributes

        else:
            feats = np.load(join(self.opt.dataset.dataset_id_path,
                                 'feats_' + flag_q + 'activation.npy'))
            if flag_q:
                questions = np.load(join(self.opt.dataset.dataset_id_path,
                                         'questions_' + flag_q + 'activation.npy'))
            else:
                raise NotImplementedError('VQA option is not available yet')
            answers = np.zeros((questions.shape[0]))
            attributes = np.load(join(self.opt.dataset.dataset_id_path,
                                      'attributes_' + flag_q + 'activation.npy'))
            return flag_q, questions, feats, answers, attributes

    def compute(self,
                stem=True,
                module=False,
                classifier=False):
        if self.threshold:
            raise NotImplemented('The implementation might not be complete! Check')
            self.compute_max(stem=stem, module=module, classifier=classifier)
        else:
            self.compute_mean_std(stem=stem, module=module, classifier=classifier)

    def map_into_attribute(self, q):
        """
        This function maps the question (one among the 21 attribute instances)
        into the attribute family. This is necessary in the stem, but not
        elsewhere, where the modulation is doing the job of selecting
        one of the 21 attributes
        :param q: question is an integer
        """
        if self.opt.dataset.experiment_case == 1:
            # experiment_case 1 is vqa
            if self.opt.hyper_method.separated_stem:
                # separate stem, else we do not do anything
                if self.opt.hyper_method.use_module == 'find':
                    if q in category_idx:
                        return 0
                    elif q in color_idx:
                        return 1
                    elif q in brightness_idx:
                        return 2
                    elif q in size_idx:
                        return 3
                    else:
                        raise ValueError('q does not fall in any attribute type')
                elif self.opt.hyper_method.use_module == 'residual':
                    return q
                else:
                    raise ValueError('Not implemented')
            else:
                return 0

    def compute_mean_std(self, stem=True, module=False, classifier=False):
        """
        compute mean activation and std
        :param stem: bool
        :param module: bool
        :param classifier: bool
        :return:
        """
        flag_q, flag_d, questions, feats, answers, attributes = self.load_data()
        type_ = int if self.approx else bool

        if self.approx:
            # marginals: we count the amount of times we see each attribute
            if self.opt.dataset.experiment_case == 0:
                factor_mean_std_s = np.zeros(len(self.vocab_q), dtype=type_)  # count
                factor_mean_std_m = np.zeros(len(self.vocab_q), dtype=type_)  # count
                factor_mean_std_c = [np.zeros(len(self.vocab_q), dtype=type_) for ccc in range(4)]  # count
            elif self.opt.dataset.experiment_case == 1:
                # 21 attributes - 21 possible questions
                factor_mean_std_m = np.zeros((len(self.vocab_q), len(self.vocab_q)), dtype=type_)  # count

        # useful for the counting the combinations
        check_hypercube = np.zeros((len(category_idx),
                                    len(color_idx),
                                    len(brightness_idx),
                                    len(size_idx)), dtype=type_)

        # load dataset -- do not change the batch size
        # critical for exact computation of weights
        dataset = DataLoaderActivations(feats, questions, answers)
        list_attributes_stem = False
        list_attribute_module = False
        list_attribute_class = False

        # for each batch
        for i_, batch in enumerate(dataset):
            (feats, questions, answers) = batch
            if isinstance(questions, list):
                questions = questions[0]
            questions_var = Variable(questions.to(device))
            questions_npy = questions.cpu().detach().numpy()

            feats_var = Variable(feats.to(device))
            self.ee(feats_var, questions_var)
            # we evaluate

            # attributes for these indexes
            tmp_attributes = attributes[dataset.batch_size*i_:dataset.batch_size*(i_+1)]
            if self.approx:  # map each element in the hypercube
                indexes_hypercube = np.apply_along_axis(self.map_into_combination, 1,
                                                        tmp_attributes)

            else:  # map for entire batch (same attributes across)
                indexes_hypercube = self.map_into_combination(attributes[dataset.batch_size*i_])

            # STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM - STEM
            if stem:
                if isinstance(self.ee.activity_stem, list):
                    # if it is a list -- specialized multitask -- stack
                    list_attributes_stem = True
                    tmp_activations_s = torch.stack(self.ee.activity_stem, dim=1).cpu().detach().numpy()
                else:
                    tmp_activations_s = self.ee.activity_stem.cpu().detach().numpy()

                # compress the spatial dimension
                tmp_activations_s = np.mean(tmp_activations_s, axis=(-2, -1)).squeeze()

                if self.approx:   # approximation to test data
                    # first iteration:
                    #       we define main variables
                    if i_ == 0:
                        if self.opt.dataset.experiment_case == 0:
                            # attributes -- statistics -- (specialized units, number of filters)
                            activations_s = np.zeros((len(self.vocab_q),) +  # attribute types
                                                     (4,) +  # statistics
                                                      tmp_activations_s.shape[1:])  # we avoid the #samples
                        elif self.opt.dataset.experiment_case == 1:
                            # size is (N, C)
                            if self.opt.hyper_method.separated_stem:
                                # specialization per type
                                if self.opt.hyper_method.use_module == 'find':
                                    factor_mean_std_s = np.zeros((len(self.vocab_q),)  # for each attribute
                                                                 + (4, ),  # count elements for each attr type
                                                                 dtype=type_)  # count
                                    activations_s = np.zeros((len(self.vocab_q),) +  # attribute types
                                                             (4,) +  # statistics
                                                             (4,) +  # different stems
                                                             tmp_activations_s.shape[1:])  # filters
                                # specialization per instance
                                elif self.opt.hyper_method.use_module == 'residual':
                                    factor_mean_std_s = np.zeros((len(self.vocab_q),  # for each attribute
                                                                  len(self.vocab_q)), dtype=type_)  # for each stem
                                    activations_s = np.zeros((len(self.vocab_q),)  # attribute
                                                             + (4,) +
                                                             (len(self.vocab_q), ) +  # different stems
                                                             tmp_activations_s.shape[1:])
                                else:
                                    raise ValueError('module not accepted')
                            else:
                                # shared stem, single module
                                factor_mean_std_s = np.zeros((len(self.vocab_q),) +  # attributes
                                                             (1,), dtype=type_)  # single stem
                                activations_s = np.zeros((len(self.vocab_q),) +  # attributes
                                                         (4,) +  # statistics
                                                         (1,) +  # stems
                                                         tmp_activations_s.shape[1:])  # filters

                    if self.opt.dataset.experiment_case == 0:   # multitask
                        for id_ex, (q_, tmp_act, tmp_tuple) in enumerate(zip(questions_npy, tmp_activations_s, tmp_attributes)):
                            for el_tuple in tmp_tuple[:4]:  # to avoid contour
                                # is there a new maximum?
                                mask_ = np.zeros(activations_s.shape, dtype=bool)  # all False
                                mask_[self.vocab_q[el_tuple], -1] = True  # last entry of stats is the max value
                                mask_act = activations_s[self.vocab_q[el_tuple], -1] < tmp_act
                                mask_[self.vocab_q[el_tuple], -1] *= mask_act
                                activations_s[mask_] = tmp_act[mask_act]

                                existing_aggregate = (factor_mean_std_s[self.vocab_q[el_tuple]],
                                                      activations_s[self.vocab_q[el_tuple], 0, :],  # mean
                                                      activations_s[self.vocab_q[el_tuple], 1, :])  # **2

                                new_aggregate = update_values(existing_aggregate,
                                                              tmp_act)
                                factor_mean_std_s[self.vocab_q[el_tuple]] = new_aggregate[0]  # N - counts
                                activations_s[self.vocab_q[el_tuple], 0, :] = new_aggregate[1]  # mean
                                activations_s[self.vocab_q[el_tuple], 1, :] = new_aggregate[2]  # M2

                    elif self.opt.dataset.experiment_case == 1:
                        # for each question in the batch we map into the branch
                        idx_stem = np.array([self.map_into_attribute(qqq) for qqq in questions_npy])
                        # for each example, we need to consider the question
                        for id_ex, (q_, id_st_, tmp_act, tmp_tuple) in enumerate(zip(questions_npy, idx_stem,
                                                                                     tmp_activations_s, tmp_attributes)):
                            for el_tuple in tmp_tuple[:4]:  # we iterate on each attribute

                                mask_ = np.zeros(activations_s.shape, dtype=bool)  # all False
                                mask_[self.vocab_q[el_tuple], -1, id_st_] = True  # last
                                mask_act = activations_s[self.vocab_q[el_tuple], -1, id_st_] < tmp_act
                                mask_[self.vocab_q[el_tuple], -1, id_st_] *= mask_act
                                activations_s[mask_] = tmp_act[mask_act]
                                # mask_ = activations_s[self.vocab_q[el_tuple]] < tmp_act
                                existing_aggregate = (factor_mean_std_s[self.vocab_q[el_tuple], id_st_],
                                                      activations_s[self.vocab_q[el_tuple], 0, id_st_],
                                                      activations_s[self.vocab_q[el_tuple], 1, id_st_])
                                # print(existing_aggregate)
                                new_aggregate = update_values(existing_aggregate,
                                                              tmp_act)
                                factor_mean_std_s[self.vocab_q[el_tuple], id_st_] = new_aggregate[0]  # we count
                                activations_s[self.vocab_q[el_tuple], 0, id_st_] = new_aggregate[1]  # mean
                                activations_s[self.vocab_q[el_tuple], 1, id_st_] = new_aggregate[2]  # M2

                else:
                    tmp_activations_s = np.mean(tmp_activations_s, axis=0).squeeze()
                    if i_ == 0:
                        activations_s = np.zeros(tuple(self.size_hypercube) + tmp_activations_s.shape)
                    activations_s[tuple(indexes_hypercube)] = tmp_activations_s
                    check_hypercube[tuple(indexes_hypercube)] = True

            # MODULE - MODULE - MODULE - MODULE - MODULE - MODULE - MODULE - MODULE - MODULE - MODULE - MODULE
            if module:
                if 'h_list' in self.ee.__dict__.keys():
                    list_attribute_module = True
                    tmp_activations_m = torch.stack(self.ee.h_list, dim=1).cpu().detach().numpy()  # multitask case
                else:
                    tmp_activations_m = self.ee.h.cpu().detach().numpy()
                tmp_activations_m = np.mean(tmp_activations_m, axis=(-2, -1)).squeeze()  # spatial average

                if self.approx:
                    if i_ == 0:  # initialization
                        if self.opt.dataset.experiment_case == 0:
                            activations_m = np.zeros((len(self.vocab_q),)  # attributes
                                                     + (4,) +  # stats
                                                     tmp_activations_m.shape[1:])  # we avoid the #samples
                        elif self.opt.dataset.experiment_case == 1:
                            activations_m = np.zeros((len(self.vocab_q),) +  # attributes
                                                     (4,) +  # stats
                                                     (len(self.vocab_q),) +  # questions -- here modulation
                                                     tmp_activations_m.shape[1:])  # we avoid the #samples

                    for id_ex, (q_, tmp_act, tmp_tuple) in enumerate(zip(questions_npy,
                                                                         tmp_activations_m,
                                                                         tmp_attributes)):
                        for el_tuple in tmp_tuple[:4]:
                            mask_ = np.zeros(activations_m.shape, dtype=bool)  # all False

                            if self.opt.dataset.experiment_case == 0:
                                # what is the new max
                                mask_[self.vocab_q[el_tuple], -1] = True  # last
                                mask_act = activations_m[self.vocab_q[el_tuple], -1] < tmp_act
                                mask_[self.vocab_q[el_tuple], -1] *= mask_act
                                activations_m[mask_] = tmp_act[mask_act]

                                existing_aggregate = (factor_mean_std_m[self.vocab_q[el_tuple]],
                                                      activations_m[self.vocab_q[el_tuple], 0, :],
                                                      activations_m[self.vocab_q[el_tuple], 1, :])
                                # print(existing_aggregate)
                                new_aggregate = update_values(existing_aggregate,
                                                              tmp_act)
                                factor_mean_std_m[self.vocab_q[el_tuple]] = new_aggregate[0]  # we count
                                activations_m[self.vocab_q[el_tuple], 0, :] = new_aggregate[1]  # mean
                                activations_m[self.vocab_q[el_tuple], 1, :] = new_aggregate[2]  # M2

                            elif self.opt.dataset.experiment_case == 1:
                                mask_[self.vocab_q[el_tuple], -1, q_] = True
                                mask_act = activations_m[self.vocab_q[el_tuple], -1, q_] < tmp_act
                                mask_[self.vocab_q[el_tuple], -1, q_] *= mask_act
                                activations_m[mask_] = tmp_act[mask_act]
                                existing_aggregate = (factor_mean_std_m[self.vocab_q[el_tuple], q_],
                                                      activations_m[self.vocab_q[el_tuple], 0, q_],
                                                      activations_m[self.vocab_q[el_tuple], 1, q_])
                                new_aggregate = update_values(existing_aggregate,
                                                              tmp_act)
                                factor_mean_std_m[self.vocab_q[el_tuple], q_] = new_aggregate[0]  # we count
                                activations_m[self.vocab_q[el_tuple], 0, q_] = new_aggregate[1]  # mean
                                activations_m[self.vocab_q[el_tuple], 1, q_] = new_aggregate[2]  # M2

                else:
                    tmp_activations_m = np.mean(tmp_activations_m, axis=0)
                    if i_ == 0:  # initialization
                        activations_m = np.zeros(tuple(self.size_hypercube) + tmp_activations_m.shape)
                    activations_m[tuple(indexes_hypercube)] = tmp_activations_m

            # CLASSIFIER - CLASSIFIER - CLASSIFIER - CLASSIFIER - CLASSIFIER - CLASSIFIER - CLASSIFIER - CLASSIFIER
            if classifier:
                if 'activity_classifier' in self.ee.__dict__.keys():
                    if isinstance(self.ee.activity_classifier, list) and self.opt.dataset.experiment_case == 0:
                        list_attribute_class = True
                    elif self.opt.dataset.experiment_case == 1:
                        list_attribute_class = False
                    else:
                        raise ValueError('Class activation not a list')

                    if self.opt.dataset.experiment_case == 0:
                        # for multitask we have a list, due to the size of amount of different classes per attr
                        if self.approx:
                            if i_ == 0:
                                activations_c = [np.zeros((len(self.vocab_q),) + (4,) + (n_clas,))
                                                 for n_clas in self.size_hypercube]  # attributes class

                            for id_class, act_class_type in enumerate(self.ee.activity_classifier):
                                tmp_activations_c = act_class_type.cpu().detach().numpy()  # for each attr type
                                for id_ex, (tmp_act, tmp_tuple) in enumerate(zip(tmp_activations_c,
                                                                                 tmp_attributes)):
                                    for el_tuple in tmp_tuple[:4]:
                                        mask_ = np.zeros(activations_c[id_class].shape, dtype=bool)  # all False
                                        mask_[self.vocab_q[el_tuple], -1] = True
                                        mask_act = activations_c[id_class][self.vocab_q[el_tuple], -1] < tmp_act
                                        mask_[self.vocab_q[el_tuple], -1] *= mask_act
                                        activations_c[id_class][mask_] = tmp_act[mask_act]

                                        existing_aggregate = (factor_mean_std_c[id_class][self.vocab_q[el_tuple]],
                                                              activations_c[id_class][self.vocab_q[el_tuple], 0, :],
                                                              activations_c[id_class][self.vocab_q[el_tuple], 1, :])
                                        new_aggregate = update_values(existing_aggregate,
                                                                      tmp_act)
                                        factor_mean_std_c[id_class][self.vocab_q[el_tuple]] = new_aggregate[0]  # we count
                                        activations_c[id_class][self.vocab_q[el_tuple], 0, :] = new_aggregate[1]  # mean
                                        activations_c[id_class][self.vocab_q[el_tuple], 1, :] = new_aggregate[2]  # M2

                        else:  #  multitask -- no approximation
                            tmp_activations_c = [np.mean(class_.cpu().detach().numpy(), axis=0)
                                                for class_ in self.ee.activity_classifier]
                            if i_ == 0:  # initialization
                                activations_c = [np.zeros(tuple(self.size_hypercube) + tmp_act_class_.shape)
                                                for tmp_act_class_ in tmp_activations_c]

                            for ccc in range(len(tmp_activations_c)):
                                activations_c[ccc][tuple(indexes_hypercube)] = tmp_activations_c[ccc]

                    elif self.opt.dataset.experiment_case == 1:
                        tmp_activations_c = self.ee.activity_classifier.cpu().detach().numpy()  # for each attr type

                        if i_ == 0:
                            if self.opt.hyper_method.separated_classifier:
                                if self.opt.hyper_method.use_module == 'find':
                                    factor_mean_std_c = np.zeros((len(self.vocab_q), 4))
                                    activations_c = np.zeros((len(self.vocab_q),) + (4,) + (4,) + tmp_activations_c.shape[1:])
                                elif self.opt.hyper_method.use_module == 'residual':
                                    factor_mean_std_c = np.zeros((len(self.vocab_q), len(self.vocab_q)))
                                    activations_c = np.zeros((len(self.vocab_q),) + (4,) + (len(self.vocab_q),) + tmp_activations_c.shape[1:])
                                else:
                                    raise ValueError('Not implemented')
                            else:
                                factor_mean_std_c = np.zeros((len(self.vocab_q), 1))
                                activations_c = np.zeros((len(self.vocab_q),) + (4,) + (1,) + tmp_activations_c.shape[1:])

                        # activations_c = np.zeros(((len(self.vocab_q),) + (4,) + (len(self.vocab_q),) + (2,)))
                        # mean, var, sample var, maximum activation
                        for id_ex, (q_, tmp_act, tmp_tuple) in enumerate(zip(questions_npy, tmp_activations_c, tmp_attributes)):
                            for el_tuple in tmp_tuple[:4]:  # for elements
                                mask_ = np.zeros(activations_c.shape, dtype=bool)
                                mask_[self.vocab_q[el_tuple], -1, self.map_into_attribute(q_)] = True
                                mask_act = activations_c[self.vocab_q[el_tuple], -1, self.map_into_attribute(q_)] < tmp_act
                                mask_[self.vocab_q[el_tuple], -1, self.map_into_attribute(q_)] *= mask_act
                                activations_c[mask_] = tmp_act[mask_act]  # we save the max here

                                existing_aggregate = (factor_mean_std_c[self.vocab_q[el_tuple], self.map_into_attribute(q_)],
                                                      activations_c[self.vocab_q[el_tuple], 0, self.map_into_attribute(q_)],
                                                      activations_c[self.vocab_q[el_tuple], 1, self.map_into_attribute(q_)])
                                new_aggregate = update_values(existing_aggregate,
                                                              tmp_act)
                                factor_mean_std_c[self.vocab_q[el_tuple], self.map_into_attribute(q_)] = new_aggregate[0]  # we count
                                activations_c[self.vocab_q[el_tuple], 0, self.map_into_attribute(q_)] = new_aggregate[1]  # mean
                                activations_c[self.vocab_q[el_tuple], 1, self.map_into_attribute(q_)] = new_aggregate[2]  # M2
                    else:
                        raise ValueError('Experiment case does not exist')

                else:
                    raise NotImplementedError('Classifier not list type does not exists')

        if self.approx:
            if stem:
                if self.opt.dataset.experiment_case == 0:
                    for k_ in sorted(self.vocab_q.keys()):
                        tuple__ = (factor_mean_std_s[self.vocab_q[k_]],
                                   activations_s[self.vocab_q[k_], 0, :],
                                   activations_s[self.vocab_q[k_], 1, :])
                        mean, var, sample_var = finalize_values(tuple__)
                        activations_s[self.vocab_q[k_], 0, :] = mean
                        activations_s[self.vocab_q[k_], 1, :] = var
                        activations_s[self.vocab_q[k_], 2, :] = sample_var

                elif self.opt.dataset.experiment_case == 1:
                    for k_ in sorted(self.vocab_q.keys()):
                        for jjj in range(factor_mean_std_s.shape[1]):
                            tuple__ = (factor_mean_std_s[self.vocab_q[k_], jjj],
                                       activations_s[self.vocab_q[k_], 0, jjj],
                                       activations_s[self.vocab_q[k_], 1, jjj])
                            mean, var, sample_var = finalize_values(tuple__)
                            activations_s[self.vocab_q[k_], 0, jjj] = mean
                            activations_s[self.vocab_q[k_], 1, jjj] = var
                            activations_s[self.vocab_q[k_], 2, jjj] = sample_var

            if module:
                if self.opt.dataset.experiment_case == 0:
                    for k_ in sorted(self.vocab_q.keys()):
                        tuple__ = (factor_mean_std_m[self.vocab_q[k_]],
                                   activations_m[self.vocab_q[k_], 0, :],
                                   activations_m[self.vocab_q[k_], 1, :])
                        mean, var, sample_var = finalize_values(tuple__)
                        activations_m[self.vocab_q[k_], 0, :] = mean
                        activations_m[self.vocab_q[k_], 1, :] = var
                        activations_m[self.vocab_q[k_], 2, :] = sample_var
                elif self.opt.dataset.experiment_case == 1:
                    for k_ in sorted(self.vocab_q.keys()):
                        for jjj in range(factor_mean_std_m.shape[1]):
                            tuple__ = (factor_mean_std_m[self.vocab_q[k_], jjj],
                                       activations_m[self.vocab_q[k_], 0, jjj],
                                       activations_m[self.vocab_q[k_], 1, jjj])
                            mean, var, sample_var = finalize_values(tuple__)
                            activations_m[self.vocab_q[k_], 0, jjj] = mean
                            activations_m[self.vocab_q[k_], 1, jjj] = var
                            activations_m[self.vocab_q[k_], 2, jjj] = sample_var

            if classifier:
                if self.opt.dataset.experiment_case == 0:
                    for k_ in sorted(self.vocab_q.keys()):
                        for id_class in range(len(factor_mean_std_c)):
                            tuple__ = (factor_mean_std_c[id_class][self.vocab_q[k_]],
                                       activations_c[id_class][self.vocab_q[k_], 0, :],
                                       activations_c[id_class][self.vocab_q[k_], 1, :])
                            mean, var, sample_var = finalize_values(tuple__)
                            activations_c[id_class][self.vocab_q[k_], 0, :] = mean
                            activations_c[id_class][self.vocab_q[k_], 1, :] = var
                            activations_c[id_class][self.vocab_q[k_], 2, :] = sample_var

                elif self.opt.dataset.experiment_case == 1:
                    for k_ in sorted(self.vocab_q.keys()):
                        for jjj in range(factor_mean_std_c.shape[1]):
                            tuple__ = (factor_mean_std_c[self.vocab_q[k_], jjj],
                                       activations_c[self.vocab_q[k_], 0, jjj],
                                       activations_c[self.vocab_q[k_], 1, jjj])
                            mean, var, sample_var = finalize_values(tuple__)
                            activations_c[self.vocab_q[k_], 0, jjj] = mean
                            activations_c[self.vocab_q[k_], 1, jjj] = var
                            activations_c[self.vocab_q[k_], 2, jjj] = sample_var

        if stem:
            if self.opt.dataset.experiment_case == 0:
                if list_attributes_stem:
                    for aa in range(len(self.size_hypercube)):  # tasks
                        if self.approx:
                            np.save(join(self.activation_path, 'approx_' + flag_d + 'activations_stem_%i.npy' % aa),
                                    activations_s[:, :, aa])
                        else:
                            np.save(join(self.activation_path, 'activations_stem_%i.npy' % aa),
                                    activations_s[:, :, :, :, aa])
                else:
                    if self.approx:
                        np.save(join(self.activation_path, 'approx_' + flag_d + 'activations_stem.npy'), activations_s)

                    else:
                        np.save(join(self.activation_path, 'activations_stem.npy'), activations_s)

            elif self.opt.dataset.experiment_case == 1:
                for jjj in range(activations_s.shape[2]):
                    np.save(join(self.activation_path, 'approx_activations_stem_%i.npy' % jjj), activations_s[:, :, jjj])

            # np.save(join(self.activation_path, 'attributes_stem.npy'), attributes)
            if self.approx:
                np.save(join(self.activation_path, 'approx_hypercube_stem.npy'), check_hypercube)
            else:
                np.save(join(self.activation_path, 'hypercube_stem.npy'), check_hypercube)

        if module:
            if self.opt.dataset.experiment_case == 0:
                if list_attribute_module:
                    for aa in range(len(self.size_hypercube)):
                        if self.approx:
                            np.save(join(self.activation_path, 'approx_' + flag_d + 'activations_module_%i.npy' % aa),
                                    activations_m[:, :, aa])
                        else:
                            np.save(join(self.activation_path, 'activations_module_%i.npy' % aa),
                                    activations_m[:, :, :, :, aa])
                else:
                    if self.approx:
                        np.save(join(self.activation_path, 'approx_' + flag_d + 'activations_module.npy'), activations_m)
                    else:
                        np.save(join(self.activation_path, 'activations_module.npy'), activations_m)

            elif self.opt.dataset.experiment_case == 1:
                for jjj in range(activations_m.shape[2]):
                    np.save(join(self.activation_path, 'approx_activations_module_%i.npy' % jjj), activations_m[:, :, jjj])


        if classifier:
            if self.opt.dataset.experiment_case == 0:
                if list_attribute_class:
                    n_class = len(activations_c)
                    for cc in range(n_class):
                        if self.approx:
                            np.save(join(self.activation_path, 'approx_' + flag_d + 'activations_class_%i.npy' % cc),
                                    activations_c[cc])
                        else:
                            np.save(join(self.activation_path, 'activations_class_%i.npy' % cc),
                                    activations_c[cc])

            elif self.opt.dataset.experiment_case == 1:
                for jjj in range(activations_c.shape[2]):
                    np.save(join(self.activation_path, 'approx_activations_class_%i.npy' % jjj), activations_c[:, :, jjj])

    def compute_max(self, stem=True, module=False, classifier=False):
        flag_q, flag_d, questions, feats, answers, attributes = self.load_data()
        if not self.approx:
            raise NotImplementedError('no max wo approx')
        dataset = DataLoaderActivations(feats, questions, answers)
        list_attributes_stem = False
        list_attribute_module = False
        list_attribute_class = False
        threshold_value = 0.1

        for j_ in range(2):  # extraction of max activation / thresholding

            for i_, batch in enumerate(dataset):
                (feats, questions, answers) = batch
                if isinstance(questions, list):
                    questions = questions[0]
                questions_var = Variable(questions.to(device))
                feats_var = Variable(feats.to(device))
                self.ee(feats_var, questions_var)
                tmp_attributes = attributes[dataset.batch_size*i_:dataset.batch_size*(i_+1)]

                if stem:
                    if isinstance(self.ee.activity_stem, list):
                        list_attributes_stem = True
                        tmp_activations_s = torch.stack(self.ee.activity_stem, dim=1).cpu().detach().numpy()
                    else:
                        tmp_activations_s = self.ee.activity_stem.cpu().detach().numpy()
                    tmp_activations_s = np.mean(tmp_activations_s, axis=(-2, -1)).squeeze()

                if j_ == 0:  # extraction of max activity
                    if i_ == 0:
                        max_activation = np.zeros(((len(self.vocab_q),) + tmp_activations_s.shape[1:]))
                        print('max activation shape', max_activation.shape)

                    for id_ex, (tmp_act, tmp_tuple) in enumerate(zip(tmp_activations_s,
                                                                     tmp_attributes)):
                        # we are iterating across samples
                        # tmp_act size - # filters: 64
                        for el_tuple in tmp_tuple[:4]:  # to avoid contour
                            mask_ = np.zeros(max_activation.shape, dtype=bool)
                            mask_[self.vocab_q[el_tuple]] = True
                            tmp_mask = max_activation[self.vocab_q[el_tuple]] < tmp_act
                            mask_[self.vocab_q[el_tuple]] *= tmp_mask
                            max_activation[mask_] = tmp_act[tmp_mask]

                else:  # threshold and counting
                    if i_ == 0:
                        threshold = threshold_value * max_activation
                        above_thr = np.zeros(((len(self.vocab_q),) + tmp_activations_s.shape[1:]))

                    for id_ex, (tmp_act, tmp_tuple) in enumerate(zip(tmp_activations_s,
                                                                     tmp_attributes)):
                        # we are iterating across samples
                        # tmp_act size - # filters: 64
                        for el_tuple in tmp_tuple[:4]:  # to avoid contour
                            tmp_mask = tmp_act > threshold
                            # above_thr[self.vocab_q[el_tuple]] *= tmp_mask
