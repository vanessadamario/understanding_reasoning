# in the main we will call data_attribute
# * generate_dataset_json

import os
import sys
import argparse
from os.path import join, isfile, dirname
from runs import experiments
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_index', type=int, required=True)
parser.add_argument('--offset_index', type=int, required=False, default=0)
parser.add_argument('--n_train_per_question', type=int, required=False)
parser.add_argument('--positive_train_combinations', type=int, required=False)
parser.add_argument('--negative_train_combinations', type=int, required=False)
parser.add_argument('--positive_test_combinations', type=int, required=False)
parser.add_argument('--negative_test_combinations', type=int, required=False)
parser.add_argument('--dataset_query', type=str, required=False)
parser.add_argument('--dataset_name', type=str, required=False)
parser.add_argument('--load_model', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--test_only', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--test_oos', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--random_data_gen', action='store_false') # type=bool, required=False, default=True)
parser.add_argument('--dense', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--data_folder', type=str, required=False, default=None)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--on_validation', action='store_true') # type=bool, required=False, default=False
parser.add_argument('--output_path', type=str, required=False)  # TODO: eliminate this
parser.add_argument('--test_seen', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--new_data_path', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--new_output_path', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--data_path', type=str, required=False, default=None)
parser.add_argument('--architecture_type', type=str, required=False, default=None)
parser.add_argument('--h5_file', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--experiment_case', type=int, required=False, default=0)  # query
parser.add_argument('--exact_activations', action='store_true') # type=bool, required=False, default=False)
parser.add_argument('--module_per_subtask', action='store_true') # type=bool, required=False, default=False)


FLAGS = parser.parse_args()
print("test oos", FLAGS.test_oos)
print("dense", FLAGS.dense)

# TODO DELETE: test
# where to save and retrieve the experiments
output_path = {
    'om2': '/--path to folder /understanding_reasoning/experiment_1',
    'om': '/--path to folder /understanding_reasoning/experiment_1'}[FLAGS.host_filesystem]
# output_path = join(output_path, 'pilot_stem_modulation/')
#print(output_path)
PATH_MNIST_SPLIT = "/--path to folder /understanding_reasoning/experiment_1/data_generation/MNIST_splits"

output_path = FLAGS.output_path
print(output_path)
os.makedirs(output_path, exist_ok=True)

if FLAGS.dense and FLAGS.data_folder is None:
    raise ValueError("You must provide a folder name")


def generate_data(id):
    """ Generation/update of the json file with data details """
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets/')
    print(output_data_folder)
    if FLAGS.dense:
        from runs import data_attribute_random
        data_attribute_random.gen_dense_questions(join(output_data_folder,
                                                       FLAGS.data_folder))
        return

    if FLAGS.random_data_gen:  # we always use this condition to generate our dataset
        from runs import data_attribute_random
        print("I am in data random")
        sys.stdout.flush()
        if FLAGS.test_only:
            path_data = join(output_data_folder, FLAGS.dataset_name)
            data_attribute_random.build_out_of_sample_test(path_data,
                                                           splits_folder=PATH_MNIST_SPLIT)
        else:
            data_attribute_random.generate_data_file(output_data_folder,
                                                     FLAGS.n_train_per_question,
                                                     FLAGS.positive_train_combinations,
                                                     FLAGS.negative_train_combinations,
                                                     FLAGS.positive_test_combinations,
                                                     FLAGS.negative_test_combinations,
                                                     h5_file=FLAGS.h5_file,
                                                     splits_folder=PATH_MNIST_SPLIT,
                                                     test_seen=FLAGS.test_seen,
                                                     dataset_name=FLAGS.dataset_name
                                                     )
    else:
        from runs import data_attribute
        data_attribute.generate_data_file(output_data_folder,
                                          h5_file=FLAGS.h5_file,
                                          splits_folder=PATH_MNIST_SPLIT)


def generate_experiments(id):
    """ Generation of the experiments. """
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets')
    experiments.generate_experiments(output_path,
                                     output_data_folder)
    # change shift based on what you need


def run_test(id):
    """ Compute test accuracy for id experiment.
    """
    # OOS:
    from runs.test import check_and_test, extract_accuracy_val
    # print(FLAGS.on_validation)
    opt = experiments.get_experiment(output_path, id)
    if FLAGS.new_output_path:
        new_path = join(FLAGS.output_path, 'train_%i' % id)
        opt.output_path = new_path
        print(new_path)
    if FLAGS.new_data_path:
        opt.dataset.dataset_id_path = join(FLAGS.data_path, opt.dataset.dataset_id)
    check_and_test(opt, FLAGS.test_oos, flag_validation=True, test_seen=True)
    check_and_test(opt, FLAGS.test_oos, flag_validation=True, test_seen=False)
    check_and_test(opt, FLAGS.test_oos, flag_validation=False, test_seen=True)
    check_and_test(opt, FLAGS.test_oos, flag_validation=False, test_seen=False)
    # extract_accuracy_val(opt, oos_distribution=FLAGS.test_oos,
    # validation=FLAGS.on_validation, test_seen=FLAGS.test_seen)


def run_train(id):
    """ Run the experiments.
    :param id: id of the experiment
    """
    import json
    from runs.train import check_and_train
    from pathlib import Path
    with open(output_path + '/train.json', 'r') as f:
        d = json.load(f)
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    if(FLAGS.load_model == True):
        fname = output_path +'/train_%d' %id + '/model.json'
        fp = Path(fname)
        if not fp.exists():
            FLAGS.load_model = False
        fname = output_path +'/flag_completed/complete_%d.txt' %id
        fp = Path(fname)
        if fp.exists():
            print("Experiment has completed!")
            return
    print("Load model at train: ", FLAGS.load_model)
    check_and_train(opt, output_path, FLAGS.load_model,
                    FLAGS.module_per_subtask)


def update_json(id):
    from runs.update import check_update
    """ Write on the json if the experiments are completed,
    by changing the flag. """
    check_update(FLAGS.output_path)


def generate_query(id):
    from runs.data_attribute_random import build_mapping
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets',
                              FLAGS.dataset_query)
    build_mapping(output_data_folder, FLAGS.test_seen)


def convert_h5_to_np(id):
    from runs.convert_hdf5 import hdf5_to_single_numpy
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets',
                              FLAGS.dataset_name)
    hdf5_to_single_numpy(id, output_data_folder, FLAGS.test_seen)


def measure_activity(id):
    path_data_activity = join(dirname(FLAGS.output_path),
                              'data_generation/datasets',
                              FLAGS.dataset_name)
    # FIXME: query case only
    # if isfile(join(path_data_activity, 'query_in_distr_indexes_activations.npy')):
    from runs.analysis_neural_activity import compute_activations
    from runs.dataset_neural_activity import generate_activation_dataset

    if FLAGS.experiment_case == 0:
        if FLAGS.exact_activations:
            if not isfile(join(path_data_activity,
                               'feats_query_activation.npy')):
                generate_activation_dataset(join(dirname(FLAGS.output_path), 'data_generation/datasets', FLAGS.dataset_name), experiment_case=0)
    compute_activations(FLAGS.architecture_type,
                        FLAGS.dataset_name,
                        FLAGS.output_path,
                        FLAGS.experiment_case,
                        FLAGS.exact_activations,
                        FLAGS.new_output_path,
                        FLAGS.new_data_path)


switcher = {
    'train': run_train,
    'gen_data': generate_data,
    'gen_exp': generate_experiments,
    'update': update_json,
    'test': run_test,
    'gen_query': generate_query,
    'activations': measure_activity,
    'convert': convert_h5_to_np
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)