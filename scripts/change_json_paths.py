# job_submission.py COPYRIGHT Fujitsu Limited 2021 and FUJITSU LABORATORIES LTD. 2021
# Authors: Atsushi Kajita (kajita@fixstars.com), G R Ramdas Pillai (ramdas@fixstars.com)

import json
import pathlib
import os
import ntpath


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

cwd = os.path.abspath(os.getcwd())

with open('results/train.json', 'r') as f:
    d = json.load(f)

for tag in d:
    d[tag]['output_path'] = cwd + '/results/train_' + tag
    d[tag]['dataset']['dataset_id_path'] = cwd + '/data_generation/datasets/' + path_leaf(d[tag]['dataset']['dataset_id_path'])

with open('results/train.json', 'w') as f:
    json.dump(d, f, indent=4)

