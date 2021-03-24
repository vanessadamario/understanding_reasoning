# job_submission.py COPYRIGHT Fujitsu Limited 2021 and FUJITSU LABORATORIES LTD. 2021
# Authors: Atsushi Kajita (kajita@fixstars.com), G R Ramdas Pillai (ramdas@fixstars.com)

import glob
import json
from pathlib import Path

lis = glob.glob('results/train_*')

for i in lis:
    with open(i + '/model.json', 'r') as f:
        d = json.load(f)
    if(d['model_t'] != 200000):
        print(i)
    
