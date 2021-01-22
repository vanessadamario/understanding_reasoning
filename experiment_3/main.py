# in the main we will call data_attribute
# * generate_dataset_json

import os
import sys
import argparse
from os.path import join
# from runs import experiments
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--offset_index', type=int, required=False, default=0)
parser.add_argument('--n_train_per_question', type=int, required=False)
parser.add_argument('--train_combinations', type=int, required=False)
parser.add_argument('--test_combinations', type=int, required=False, default=5)
parser.add_argument('--dataset_name', type=str, required=False)
parser.add_argument('--load_model', type=bool, required=False, default=False)
parser.add_argument('--random_data_gen', type=bool, required=False, default=True)
parser.add_argument('--data_folder', type=str, required=False, default=None)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)

FLAGS = parser.parse_args()
# where to save and retrieve the experiments
output_path = {
    'om': '/om/user/vanessad/understanding_reasoning/experiment_3',
    'om2': '/om2/user/vanessad/understanding_reasoning/experiment_3',
    'vanessa': '/Users/vanessa/src/understanding_reasoning/experiment_3'}[FLAGS.host_filesystem]
output_path = join(output_path, 'results/')
print(output_path)
PATH_MNIST_SPLIT = "/om2/user/vanessad/understanding_reasoning/experiment_3/data_generation/MNIST_splits"
os.makedirs(output_path, exist_ok=True)


def generate_data(id):
    """ Generation/update of the json file with data details """
    from runs import data_attribute_random
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets/')
    data_attribute_random.generate_data_file(output_data_folder,
                                             FLAGS.n_train_per_question,
                                             FLAGS.train_combinations,
                                             FLAGS.test_combinations,
                                             splits_folder=PATH_MNIST_SPLIT)


def generate_experiments(id):
    """ Generation of the experiments. """
    from runs import experiments
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets')
    experiments.generate_experiments(output_path,
                                     output_data_folder)


def run_test(id):
    """ Compute test accuracy for id experiment.
    """
    # OOS:
    from runs.test import check_and_test
    print(FLAGS.test_oos)
    opt = experiments.get_experiment(output_path, id)
    check_and_test(opt, FLAGS.test_oos)


def run_train(id):
    """ Run the experiments.
    :param id: id of the experiment
    """
    # TODO: use mostly the functions and models from systematic generalization
    from runs.train import check_and_train
    from runs import experiments
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    check_and_train(opt, output_path, FLAGS.load_model)


def update_json(id):
    from runs.update import check_update
    """ Write on the json if the experiments are completed,
    by changing the flag. """
    print(output_path)
    check_update(output_path)


switcher = {
    'train': run_train,
    'gen_data': generate_data,
    'gen_exp': generate_experiments,
    'update': update_json,
    'test': run_test,
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)