# in the main we will call data_attribute
# * generate_dataset_json

import os
import argparse
from os.path import join
from runs import experiments
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--relations', type=bool, required=False, default=True)
parser.add_argument('--attribute_comparison', type=bool, required=False, default=True)
parser.add_argument('--offset_index', type=int, required=False, default=0)
parser.add_argument('--variety', type=int, required=False)
parser.add_argument('--split', type=list, required=False, default=None)
parser.add_argument('--output_folder', type=str, required=False, default="results")
parser.add_argument('--dataset_name', type=str, required=False, default=None)
parser.add_argument('--load_model', type=bool, required=False, default=False)
parser.add_argument('--root_data_folder', type=str, required=False, default=None)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--test_oos', type=bool, required=False, default=False)
parser.add_argument('--test_seen', type=bool, required=False, default=False)
parser.add_argument('--on_validation', type=bool, required=False, default=False)
parser.add_argument('--h5_file', type=bool, required=False, default=False)
parser.add_argument('--run', type=str, required=True)

FLAGS = parser.parse_args()
# where to save and retrieve the experiments
output_path = {
    'om': '/om/user/vanessad/understanding_reasoning/experiment_4',
    'om2_exp4': '/om2/user/vanessad/understanding_reasoning/experiment_4',
    'om2_exp2': '/om2/user/vanessad/understanding_reasoning/experiment_2',
    'vanessa': '/Users/vanessa/src/understanding_reasoning/experiment_4'}[FLAGS.host_filesystem]
output_path = join(output_path, FLAGS.output_folder + "/")
print("output path: %s" % output_path)
PATH_MNIST_SPLIT = "/om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/MNIST_splits"
os.makedirs(output_path, exist_ok=True)
print("Load model: ", FLAGS.load_model)


def generate_data(id):
    """ Generation/update of the json file with data details """
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets/')
    print(output_data_folder)
    output_data_path = join(output_data_folder, FLAGS.dataset_name)
    os.makedirs(output_data_path, exist_ok=True)

    if FLAGS.relations:
        if FLAGS.attribute_comparison:
            from runs.data_comparison_relations import DataGenerator
            if FLAGS.host_filesystem == 'om2_exp4':
                DG = DataGenerator(PATH_MNIST_SPLIT,
                                   variety=FLAGS.variety,
                                   image_size=64,
                                   gen_in_distr_test=FLAGS.test_seen)
            elif FLAGS.host_filesystem == 'om2_exp2':
                from runs.data_comparison_relations import RELATIONS_DICT_EXP2
                print(RELATIONS_DICT_EXP2)
                DG = DataGenerator(PATH_MNIST_SPLIT,
                                   variety=FLAGS.variety,
                                   image_size=28,
                                   relations=RELATIONS_DICT_EXP2,
                                   single_image=False,
                                   gen_in_distr_test=FLAGS.test_seen)

            else:
                raise ValueError("Data generation protocol does not exist")

        else:
            from runs.data_attribute_random import DataGenerator
            DG = DataGenerator(PATH_MNIST_SPLIT,
                               variety=FLAGS.variety)

        DG.generate_data_matrix(savepath=output_data_path, h5_file=FLAGS.h5_file)


def generate_experiments(id):
    """ Generation of the experiments. """
    if FLAGS.root_data_folder is 'sqoop':
        output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                                  'data_generation/sysgen_sqoop')
    else:
        output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                                  'data_generation/datasets/')

    experiments.generate_experiments(output_path,
                                     output_data_folder)


def run_test(id):
    """ Compute test accuracy for id experiment.
    """
    from runs.test import check_and_test, extract_accuracy_val
    print(FLAGS.test_oos)
    opt = experiments.get_experiment(output_path, id)
    # flag_validation has priority on test if both true

    check_and_test(opt, FLAGS.test_oos, flag_validation=True, test_seen=True)
    check_and_test(opt, FLAGS.test_oos, flag_validation=True, test_seen=False)
    check_and_test(opt, FLAGS.test_oos, flag_validation=False, test_seen=True)
    check_and_test(opt, FLAGS.test_oos, flag_validation=False, test_seen=False)


def run_train(id):
    """ Run the experiments.
    :param id: id of the experiment
    """
    from runs.train import check_and_train
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    check_and_train(opt, output_path, FLAGS.load_model)


def update_json(id):
    from runs.update import check_update
    """ Write on the json if the experiments are completed,
    by changing the flag. """
    check_update(output_path)


def generate_query(id):
    from runs.data_attribute_random import build_mapping
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets',  # change
                              FLAGS.dataset_query)
    build_mapping(output_data_folder)


def convert_h5_to_np(id):
    from runs.convert_hdf5 import hdf5_to_single_numpy
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets',
                              FLAGS.dataset_name)
    hdf5_to_single_numpy(id, output_data_folder)


switcher = {
    'train': run_train,
    'gen_data': generate_data,
    'gen_exp': generate_experiments,
    'update': update_json,
    'test': run_test,
    'gen_query': generate_query,
    'convert': convert_h5_to_np
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)