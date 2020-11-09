# in the main we will call data_attribute
# * generate_dataset_json

import os
import argparse
from os.path import join
from runs import experiments
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--offset_index', type=int, required=False)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)

FLAGS = parser.parse_args()

# where to save and retrieve the experiments
output_path = {
    'om': '/om/user/vanessad/understanding_reasoning/experiment_1',
    'vanessa': '/Users/vanessa/src/understanding_reasoning/experiment_1'}[FLAGS.host_filesystem]
output_path = join(output_path, 'results/')
PATH_MNIST_SPLIT = "/om/user/vanessad/understanding_reasoning/experiment_1/data_generation/MNIST_splits"
os.makedirs(output_path, exist_ok=True)

if FLAGS.offset_index is None:
    FLAGS.offset_index = 0


def generate_data(id):
    """ Generation/update of the json file with data details """
    from runs import data_attribute
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets/')
    data_attribute.generate_data_file(output_data_folder,
                                      splits_folder=PATH_MNIST_SPLIT)


def generate_experiments(id):
    """ Generation of the experiments. """
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets')
    experiments.generate_experiments(output_path,
                                     output_data_folder)


def run_train(id):
    """ Run the experiments.
    :param id: id of the experiment
    """
    # TODO: use mostly the functions and models from systematic generalization
    from runs.train import check_and_train
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    check_and_train(opt, output_path)


def update_json(id):
    from runs.update import check_update
    """ Write on the json if the experiments are completed,
    by changing the flag. """
    check_update(output_path)


switcher = {
    'train': run_train,
    'gen_data': generate_data,
    'gen_exp': generate_experiments,
    'update': update_json
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)