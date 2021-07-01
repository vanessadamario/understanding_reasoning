import os
from os.path import join
import sys


def check_and_train(opt,
                    output_path,
                    load=False,
                    sqoop_dataset=False,
                    shaping=False, path_shaping=None):
    """ Check if the experiments has already been performed.
    If it is not, train otherwise retrieve the path relative to the experiment.
    :param opt: Experiment instance. It contains the output path for the experiment
    under study
    :param output_path: the output path for the *.json file. necessary to change the *.json
    :param load: bool, if False, train from scratch, else resume model
    """
    if sqoop_dataset:
        from runs.data_loader_sqoop import ClevrDataLoader #  as DataTorchLoader
    else:
        from runs.data_loader import DataTorchLoader

    print(opt)
    print("END")
    if opt.train_completed:
        print("Object: ", opt)
        print("Experiment already trained in " + opt.output_path)

    print(output_path)
    sys.stdout.flush()

    print('check and train', opt.output_path)
    sys.stdout.flush()

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    # TODO: training
    # we need to load the data
    # TODO 1: we first need to generate the data
    # we need to fix the question we want to make, per each element
    # image, question, answer, program
    # and pass them similarly to what ClevrDataset does
    # worst case to pass those to train_loop as in the original implementation

    if sqoop_dataset:
        opt_ = {'question_h5': join(opt.dataset.dataset_id_path, 'train_questions.h5'),
                'feature_h5': join(opt.dataset.dataset_id_path, 'train_features.h5'),
                'vocab': join(opt.dataset.dataset_id_path, 'vocab.json'),
                'batch_size': opt.hyper_opt.batch_size}
        train_loader = ClevrDataLoader(**opt_)

        opt_ = {'question_h5': join(opt.dataset.dataset_id_path, 'val_questions.h5'),
                'feature_h5': join(opt.dataset.dataset_id_path, 'val_features.h5'),
                'vocab': join(opt.dataset.dataset_id_path, 'vocab.json'),
                'batch_size': opt.hyper_opt.batch_size}
        valid_loader = ClevrDataLoader(**opt_)

    else:
        train_loader = DataTorchLoader(opt)  # at training
        for tr_ in train_loader:  # TODO check here
            print(tr_[0].shape, tr_[1], tr_[2])
            print(tr_[0].ndimension())
            break

        valid_loader = DataTorchLoader(opt, split="valid")
    # TODO 2: we need to call the train_loop function

    from runs.train_loop import train_loop
    train_loop(opt, train_loader, valid_loader, load, shaping, path_shaping)
    # here training must happen

    # we write an empty *.txt file with the completed experiment
    flag_completed_dir = join(output_path, 'flag_completed')
    os.makedirs(flag_completed_dir, exist_ok=True)
    file_object = open(join(flag_completed_dir, "complete_%s.txt" % str(opt.id)), "w")
    file_object.close()