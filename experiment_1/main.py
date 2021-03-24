# in the main we will call data_attribute
# * generate_dataset_json

import os
import sys
import argparse
from os.path import join
from runs import experiments
from pathlib import Path

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
parser.add_argument('--load_model', type=bool, required=False, default=False)
parser.add_argument('--test_only', type=bool, required=False, default=False)
parser.add_argument('--test_oos', type=bool, required=False, default=False)
parser.add_argument('--random_data_gen', type=bool, required=False, default=True)
parser.add_argument('--dense', type=bool, required=False, default=False)
parser.add_argument('--data_folder', type=str, required=False, default=None)
parser.add_argument('--host_filesystem', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--on_validation', type=bool, required=False, default=False)

FLAGS = parser.parse_args()
print("test oos", FLAGS.test_oos)
print("dense", FLAGS.dense)

# where to save and retrieve the experiments
output_path = join(Path(__file__).parent.absolute(), 'results/')
PATH_MNIST_SPLIT = "/om2/user/vanessad/understanding_reasoning/experiment_1/data_generation/MNIST_splits"

print(output_path)
os.makedirs(output_path, exist_ok=True)

if FLAGS.dense and FLAGS.data_folder is None:
    raise ValueError("You must provide a folder name")


def generate_data(id):
    """ Generation/update of the json file with data details """
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets/')
    if FLAGS.dense:
        from runs import data_attribute_random
        data_attribute_random.gen_dense_questions(join(output_data_folder,
                                                       FLAGS.data_folder))
        return

    if FLAGS.random_data_gen:
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
                                                     splits_folder=PATH_MNIST_SPLIT)
    else:
        from runs import data_attribute
        data_attribute.generate_data_file(output_data_folder,
                                          splits_folder=PATH_MNIST_SPLIT)


def generate_experiments(id):
    """ Generation of the experiments. """
    output_data_folder = join(os.path.dirname(os.path.dirname(output_path)),
                              'data_generation/datasets')
    # output_data_folder = "/om/user/vanessad/understanding_reasoning/experiment_1/data_generation/datasets"
    # output_path = "/om5/user/vanessad/understanding_reasoning/experiment_1/results"
    experiments.generate_experiments(output_path,
                                     output_data_folder)


def run_test(id):
    """ Compute test accuracy for id experiment.
    """
    # OOS:
    from runs.test import check_and_test, extract_accuracy_val
    print(FLAGS.on_validation)
    opt = experiments.get_experiment(output_path, id)
    check_and_test(opt, FLAGS.test_oos, flag_validation=FLAGS.on_validation)
    extract_accuracy_val(opt, oos_distribution=FLAGS.test_oos, validation=FLAGS.on_validation)


def run_train(id):
    """ Run the experiments.
    :param id: id of the experiment
    """
    # TODO: use mostly the functions and models from systematic generalization
    from runs.train import check_and_train
    from pathlib import Path
    import json
    import os
    opt = experiments.get_experiment(output_path, id)  # Experiment instance
    with open('results/train.json', 'r') as f:
        d = json.load(f)

    if not os.path.exists(d[str(id)]['dataset']['dataset_id_path']):
        return

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
                              'data_generation/datasets',
                              FLAGS.dataset_query)
    build_mapping(output_data_folder)


switcher = {
    'train': run_train,
    'gen_data': generate_data,
    'gen_exp': generate_experiments,
    'update': update_json,
    'test': run_test,
    'gen_query': generate_query
}

switcher[FLAGS.run](FLAGS.experiment_index + FLAGS.offset_index)
