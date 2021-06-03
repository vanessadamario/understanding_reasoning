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
parser.add_argument('--output_folder', type=str, required=False, default="results")  # aws
parser.add_argument('--dataset_name', type=str, required=False, default=None)
parser.add_argument('--single_image', type=bool, required=False, default=True)
parser.add_argument('--spatial_only', type=bool, required=False, default=False)
parser.add_argument('--load_model', type=bool, required=False, default=False)
parser.add_argument('--root_data_folder', type=str, required=False, default=None)  # also AWS
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--test_oos', type=bool, required=False, default=False)
parser.add_argument('--test_seen', type=bool, required=False, default=False)
parser.add_argument('--on_validation', type=bool, required=False, default=False)
parser.add_argument('--h5_file', type=bool, required=False, default=False)
parser.add_argument('--modify_path', type=bool, required=False, default=False)  # AWS experiments
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--data_path', type=str, required=False, default=None)


FLAGS = parser.parse_args()
print(FLAGS)
# where to save and retrieve the experiments
output_path = {
    'om2_exp4': '--path to folder /understanding_reasoning/experiment_4',
    'om2_exp2': '--path to folder /understanding_reasoning/experiment_2'}[FLAGS.host_filesystem]
output_path = join(output_path, FLAGS.output_folder + "/")
print("output path: %s" % output_path)
PATH_MNIST_SPLIT = "--path to folder /understanding_reasoning/experiment_1/data_generation/MNIST_splits"
os.makedirs(output_path, exist_ok=True)
print("Load model: ", FLAGS.load_model)


def generate_data(id):
    """ Generation/update of the json file with data details """
    if FLAGS.root_data_folder is None:
        output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                                  'data_generation/datasets/')
    else:
        output_data_folder = FLAGS.root_data_folder
    print(output_data_folder)
    output_data_path = join(output_data_folder, FLAGS.dataset_name)
    os.makedirs(output_data_path, exist_ok=True)
    print(output_data_path)
    if FLAGS.relations:
        if FLAGS.attribute_comparison:
            from runs.data_comparison_relations import DataGenerator
            if FLAGS.host_filesystem == 'om2_exp4':
                DG = DataGenerator(PATH_MNIST_SPLIT,
                                   variety=FLAGS.variety,
                                   image_size=128,  # 64 if in the old version
                                   spatial_only=FLAGS.spatial_only,
                                   single_image=True,
                                   gen_in_distr_test=FLAGS.test_seen)
            elif FLAGS.host_filesystem == 'om2_exp2':
                from runs.data_comparison_relations import RELATIONS_DICT_EXP2
                print(RELATIONS_DICT_EXP2)
                DG = DataGenerator(PATH_MNIST_SPLIT,
                                   variety=FLAGS.variety,
                                   image_size=28,
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
    if FLAGS.modify_path:
        _data_name = opt.dataset.dataset_id
        _train_id = opt.id
        opt.dataset.dataset_id_path = join(FLAGS.root_data_folder, _data_name)
        opt.output_path = join(output_path, 'train_%i' % _train_id)

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
    from pathlib import Path
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    if (FLAGS.load_model == True):
        fname = output_path + '/train_%d' % id + '/model.json'
        fp = Path(fname)
        if not fp.exists():
            FLAGS.load_model = False
        fname = output_path + '/flag_completed/complete_%d.txt' % id
        fp = Path(fname)
        if fp.exists():
            print("Experiment has completed!")
            return
    print("Load model at train: ", FLAGS.load_model)
    check_and_train(opt, output_path, FLAGS.load_model)


def update_json(id):
    from runs.update import check_update
    """ Write on the json if the experiments are completed,
    by changing the flag. """
    print(output_path)
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


def shaping(id):
    from runs.pilot_shaping import run_pilot
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    path_starting_model = run_pilot(opt)
    print(path_starting_model)
    from runs.train import check_and_train
    check_and_train(opt,
                    output_path,
                    load=FLAGS.load_model,
                    shaping=True,
                    path_shaping=path_starting_model)

switcher = {
    'train': run_train,
    'gen_data': generate_data,
    'gen_exp': generate_experiments,
    'update': update_json,
    'test': run_test,
    'gen_query': generate_query,
    'convert': convert_h5_to_np,
    'shaping': shaping
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)