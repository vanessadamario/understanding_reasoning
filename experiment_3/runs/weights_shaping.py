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
            list_stem_transf = []
            for layer in model[0].stem[k_]:
                try:
                    list_stem_transf.append(layer.weight)
                except:
                    list_stem_transf.append([])
            dict_stem[k_] = list_stem_transf
        torch.save(dict_stem,
                   join(opt.output_path, 'stem'))
    else:
        list_stem_transf = []
        for layer in model[0].stem:
            try:
                list_stem_transf.append(layer.weight)
            except:
                list_stem_transf.append([])
        torch.save(list_stem_transf,
                   join(opt.output_path, 'stem'))

