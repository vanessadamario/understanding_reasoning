#!/usr/bin/env python3

# Copyright 2019-present, Mila
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import json
import torch
import numpy as np
from os.path import join


# from vr.models import (ModuleNet,
#                        SHNMN,
#                        Seq2Seq,
#                        Seq2SeqAtt,
#                        LstmModel,
#                        CnnLstmModel,
#                        CnnLstmSaModel,
#                        FiLMedNet,
#                        FiLMGen,
#                        MAC,
#                        RelationNet)

# UTILS NOTEBOOK


def search_experiments(df, common_hypers):
    """ Find the experiments that share identical hyper-parameters
    :param df: pd.DataFrame with experiments characteristics
    :param common_hypers: dictionary of hyper-parameters,
        as for the data generation
    """
    # HYPER_PARAMS TEMPLATE :
    # here we exclude batch_size, learning_rate
    # common_hypers = {"method_type": "SHNMN",
    #                  "hyper_method": {"use_module": "find",
    #                                   "alpha_init": "single",
    #                                   "tau_init": "single",
    #                                    "feature_dim": [3,28,28]},
    #                 "dataset": {"dataset_id": "dataset_5"}}

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
    return np.squeeze(np.argwhere(same_exp_array))


def find_best_experiment(df, indexes):
    """
    :param df: pd.DataFrame
    :param indexes: indexes for experiments that are comparable
    :return:
    """
    path_list = [df["output_path"][k_] for k_ in indexes]
    max_list = []

    for id_, path_ in enumerate(path_list):
        json_output = json.load(open(join(path_, "model.json"), "rb"))
        max_list.append((np.max(np.array(json_output["val_accs"]))))
    return indexes[np.argmax(max_list)], np.max(max_list)


def invert_dict(d):
    return {v: k for k, v in d.items()}


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        print(vocab)
        vocab_copy = vocab.copy()
        for key_ in vocab.keys():
            if key_ == 'question_token_to_idx':
                vocab_copy['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
            elif key_ == 'program_token_to_idx':
                vocab_copy['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
            elif key_ == 'answer_token_to_idx':
                vocab_copy['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])

    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    #  assert vocab['question_token_to_idx']['<NULL>'] == 0
    # assert vocab['question_token_to_idx']['<START>'] == 1
    # assert vocab['question_token_to_idx']['<END>'] == 2
    # assert vocab['program_token_to_idx']['<NULL>'] == 0
    # assert vocab['program_token_to_idx']['<START>'] == 1
    # assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab_copy


def load_cpu(path):
    """
    Loads a torch checkpoint, remapping all Tensors to CPU
    """
    return torch.load(path, map_location=lambda storage, loc: storage)


def load_program_generator(path):
    from runs.shnmn import SHNMN
    checkpoint = load_cpu(path)
    model_type = checkpoint['args']['model_type']
    kwargs = checkpoint['program_generator_kwargs']
    state = checkpoint['program_generator_state']
    if model_type in ['FiLM', 'MAC', 'RelNet']:
        kwargs = get_updated_args(kwargs, FiLMGen)
        model = FiLMGen(**kwargs)
    elif model_type == 'PG+EE':
        if kwargs.rnn_attention:
            model = Seq2SeqAtt(**kwargs)
        else:
            model = Seq2Seq(**kwargs)
    else:
        model = None
    if model is not None:
        model.load_state_dict(state)
    return model, kwargs


def load_execution_engine(path,
                          model_type="SHNMN",
                          verbose=True):
    checkpoint = load_cpu(path)
    from runs.shnmn import SHNMN
    kwargs = checkpoint['execution_engine_kwargs']
    state = checkpoint['execution_engine_state']
    kwargs['verbose'] = verbose
    if model_type == 'SHNMN':
        model = SHNMN(**kwargs)
    else:
        raise ValueError()
    cur_state = model.state_dict()
    # TODO: modified
    model.load_state_dict(state)
    return model, kwargs


def load_baseline(path):
    model_cls_dict = {
        'LSTM': LstmModel,
      'CNN+LSTM': CnnLstmModel,
      'CNN+LSTM+SA': CnnLstmSaModel,
    }
    checkpoint = load_cpu(path)
    baseline_type = checkpoint['baseline_type']
    kwargs = checkpoint['baseline_kwargs']
    state = checkpoint['baseline_state']

    model = model_cls_dict[baseline_type](**kwargs)
    model.load_state_dict(state)
    return model, kwargs


def get_updated_args(kwargs, object_class):
    """
    Returns kwargs with renamed args or arg values and deleted, deprecated, unused args.
    Useful for loading older, trained models.
    If using this function is necessary, use immediately before initializing object.
    """
    # Update arg values
    for arg in arg_value_updates:
        if arg in kwargs and kwargs[arg] in arg_value_updates[arg]:
            kwargs[arg] = arg_value_updates[arg][kwargs[arg]]

    # Delete deprecated, unused args
    valid_args = inspect.getargspec(object_class.__init__)[0]
    new_kwargs = {valid_arg: kwargs[valid_arg] for valid_arg in valid_args if valid_arg in kwargs}
    return new_kwargs


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, cat, name, val):
        self.shadow[cat + '-' + name] = val.clone()

    def __call__(self, cat, name, x):
        name = cat + '-' + name
        assert name in self.shadow
        new_average = self.mu * x + (1.0 - self.mu) * self.shadow[name]
        self.shadow[name] = new_average.clone()
        return new_average


arg_value_updates = {
    'condition_method': {
        'block-input-fac': 'block-input-film',
    'block-output-fac': 'block-output-film',
    'cbn': 'bn-film',
    'conv-fac': 'conv-film',
    'relu-fac': 'relu-film',
  },
  'module_input_proj': {
      True: 1,
  },
}
