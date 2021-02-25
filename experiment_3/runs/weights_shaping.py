import os
import json
import sys
import numpy as np
import torch
import pandas as pd
from os.path import join
import inspect

from runs.shnmn import SHNMN
from runs.utils import load_vocab, load_execution_engine


def save_weights(opt):
    model = load_execution_engine(join(opt.output_path, 'model'), model_type='SHNMN')

    if opt.hyper_method.use_module == 'find':
        torch.save(model[0].func, join(opt.output_path, 'func'))

    torch.save(model[0].question_embeddings.weight,
               join(opt.output_path, 'embedding_weights'))

    if opt.hyper_method.separated_stem:
        dict_stem = {}
        for k_ in model[0].stem.keys():  # for each key (red, green, etc)
            # print('key: ', k_)
            list_stem_transf = []
            for layer in model[0].stem[k_]:
                # print(layer)
                if isinstance(layer, torch.nn.BatchNorm2d):
                    list_stem_transf.append([layer.running_mean, layer.running_var])
                elif isinstance(layer, torch.nn.Conv2d):
                    list_stem_transf.append([layer.weight, layer.bias])
                elif isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.MaxPool2d):
                    list_stem_transf.append([])
                else:
                    raise ValueError('Instance type not recognized')
            dict_stem[k_] = list_stem_transf
        torch.save(dict_stem,
                   join(opt.output_path, 'stem'))

    else:
        list_stem_transf = []
        for layer in model[0].stem:
            # print(layer)
            if isinstance(layer, torch.nn.BatchNorm2d):
                list_stem_transf.append([layer.running_mean, layer.running_var])
            elif isinstance(layer, torch.nn.Conv2d):
                list_stem_transf.append([layer.weight, layer.bias])
            elif isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.MaxPool2d):
                list_stem_transf.append([])
            else:
                raise ValueError('Instance type not recognized')
        torch.save(list_stem_transf,
                   join(opt.output_path, 'stem'))
