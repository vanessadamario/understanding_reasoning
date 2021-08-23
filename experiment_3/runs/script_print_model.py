import numpy
import sys
from os.path import join
from runs.shnmn import SHNMN
from runs.test import load_execution_engine


def print_ee(opt):
    print(opt.output_path)
    model = load_execution_engine(join(opt.output_path, 'model'), model_type='SHNMN')
    print(model)
    sys.stdout.flush()
