import numpy as np
import json
import os
from os.path import join, dirname


def rm_exps(path_train_json,
            remove_begin=18,
            remove_end=36):
    path_train_json = join(path_train_json, 'train.json')
    train_json = json.load(open(path_train_json, 'r')).copy()

    for j_ in range(remove_begin, remove_end):
        del train_json[str(j_)]

    with open(join(dirname(path_train_json), 'rm_train.json'), 'w') as outfile:
        json.dump(train_json, outfile, indent=4)


def main():
    rm_exps('/om2/user/vanessad/understanding_reasoning/experiment_4/results_NeurIPS_revision')


if __name__ == '__main__':
    main()