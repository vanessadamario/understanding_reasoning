import numpy as np
import json
from os.path import join, dirname


def change_path(path_train_json, destination='dgx1', user='vanessad'):

    if destination == 'dgx1':
        root_path_data = join('/raid/poggio/home', user, 'understanding_reasoning/experiment_4/datasets')
        root_path_results = join('/raid/poggio/home', user, 'understanding_reasoning/experiment_4/results_NeurIPS_module_per_subtask_trial2')
    else:
        raise ValueError('No other destinations exists at the moment')

    train_json = json.load(open(path_train_json, 'r')).copy()

    for j_ in train_json:
        preserve_results_path = train_json[j_]['output_path'].split('results_NeurIPS_module_per_subtask_trial2/')[-1]
        train_json[j_]['output_path'] = join(root_path_results, preserve_results_path)

        preserve_data_path_lst = train_json[j_]['dataset']['dataset_id_path'].split('datasets/')[-1]
        train_json[j_]['dataset']['dataset_id_path'] = join(root_path_data,
                                                            preserve_data_path_lst)
        # change here
        # preserve_data_path_lst = train_json[j_]['dataset']['dataset_id_path'].split('/')
        # train_json[j_]['dataset']['dataset_id_path'] = join(root_path_data,
        #                                                     preserve_data_path_lst[-4],
        #                                                     preserve_data_path_lst[-1])

    with open(join(dirname(path_train_json), 'train_%s.json' % destination), 'w') as outfile:
        json.dump(train_json, outfile, indent=4)


def main():
    # change_path('/om2/user/vanessad/reasoning_partII/results/experiment_4/shaping_trial_1/train.json')
    # change_path('/om2/user/vanessad/reasoning_partII/results/experiment_4/reduced_data/attributes_10k_shaping/train.json')
    # change_path('/om2/user/vanessad/reasoning_partII/results/experiment_4/reduced_data_loose_50k/attributes_10k_scratch/train.json')
    # change_path('/om2/user/vanessad/reasoning_partII/results/experiment_4/reduced_data_loose_50k/attributes_10k_shaping/train.json')
    change_path('/om2/user/vanessad/understanding_reasoning/experiment_4/results_NeurIPS_module_per_subtask_trial2/train.json')


if __name__ == '__main__':
    main()
