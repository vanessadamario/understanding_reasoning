import os
import sys
import pandas as pd
from os.path import join
import numpy as np
sys.path.append('/om2/user/vanessad/understanding_reasoning/experiment_3')
from runs.shnmn import SHNMN
import inspect
print('path file')
print(inspect.getfile(SHNMN))


def map_dataset(experiment_case, dataset_name):
    if experiment_case == 3:
        return dataset_name
    elif experiment_case == 1:
        if dataset_name == 'dataset_0':
            return 'dataset_15'
        elif dataset_name == 'dataset_1':
            raise ValueError('this seen amount of combinations is missing')
        elif dataset_name == 'dataset_2':
            return 'dataset_16'
        elif dataset_name == 'dataset_3':
            raise ValueError('this seen amount of combinations is missing')
        elif dataset_name == 'dataset_4':
            return 'dataset_17'
        elif dataset_name == 'dataset_5':
            return 'dataset_18'
        else:
            raise ValueError('No correspondence')


def run_pilot(opt):

   # Extract index
    # opt is for experiment 4 or 2
    if opt.hyper_method.feature_dim[1] == 28:
        experiment_case = 1   # [3,28,28], we are in experiment 2 and we want to recover experiment_1
    elif opt.hyper_method.feature_dim[1] == 64:
        experiment_case = 3
    else:
        raise ValueError('Experiment case without correspondence')

    path_df = join(os.path.dirname(os.path.dirname(os.path.dirname(opt.output_path))),
                   'experiment_%i' % experiment_case,  'results/train.json')

    df_attr = pd.read_json(path_df).T

    hyper_names = ['hyper_opt', 'hyper_method', 'dataset']
    exclude_list = [[],
                    ['num_modules', 'tau_init', 'alpha_init'],
                    ['dataset_id_path', 'image_size', 'n_training']]

    mask = np.ones(df_attr.shape[0], dtype=bool)
    # Check hyper-parameters
    for name_, exclude_ in zip(hyper_names, exclude_list):
        hyper_attr = pd.DataFrame(data=[df_attr.iloc[j_][name_].values()
                                        for j_ in range(df_attr.shape[0])],
                                  columns=df_attr.iloc[0][name_].keys())
        keys_hyper_attr = set(list(hyper_attr.columns))

        dict_hyper_comp = {}
        opt_hyper_comp = getattr(opt, name_)
        for key, val in opt_hyper_comp.__dict__.items():
            attribute = opt_hyper_comp.__getattribute__(key)
            print('%s: ' % key, attribute)
            # None values give error in pandas -- keys that we don't want to consider
            if key == 'dataset_id':
                dict_hyper_comp[key] = map_dataset(experiment_case, attribute)
            elif not (attribute is None or key in exclude_):
                dict_hyper_comp[key] = attribute

        keys_hyper_comp = set(list(dict_hyper_comp.keys()))
        union_keys = keys_hyper_attr.union(keys_hyper_comp)
        intersection_keys = keys_hyper_attr.intersection(keys_hyper_comp)

        for no_intersection in union_keys - intersection_keys:
            try:
                print(no_intersection)
                del hyper_attr[no_intersection]
            except:
                pass

        ordered_hyper_comp = [dict_hyper_comp[k_] for k_ in hyper_attr.columns]
        mask *= (hyper_attr == ordered_hyper_comp).all(1).values

    if np.where(mask)[0].size != 1:
        print('Corresponding indexes: ', np.where(mask)[0])
        raise ValueError('Size is not one')

    id_experiment = np.where(mask)[0][0]

    return join(os.path.dirname(path_df), 'train_%i' % id_experiment)
