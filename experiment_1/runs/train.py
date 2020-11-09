import os
from os.path import join
from runs.data_loader import DataTorchLoader
from runs.train_loop import train_loop


def check_and_train(opt, output_path):
    """ Check if the experiments has already been performed.
    If it is not, train otherwise retrieve the path relative to the experiment.
    :param opt: Experiment instance. It contains the output path for the experiment
    under study
    :param output_path: the output path for the *.json file. necessary to change the *.json
    """
    print(opt)
    print("END")
    if opt.train_completed:
        print("Object: ", opt)
        print("Experiment already trained in " + opt.output_path)

    print(output_path)
    print('check and train', opt.output_path)

    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    # TODO: training
    # we need to load the data
    # TODO 1: we first need to generate the data
    # we need to fix the question we want to make, per each element
    # image, question, answer, program
    # and pass them similarly to what ClevrDataset does
    # worst case to pass those to train_loop as in the original implementation
    train_loader = DataTorchLoader(opt)  # at training
    for tr_ in train_loader:
        print(tr_[0].shape, tr_[1], tr_[2])
        break
    valid_loader = DataTorchLoader(opt, split="valid")
    # TODO 2: we need to call the train_loop function
    train_loop(opt, train_loader, valid_loader)
    # here training must happen
    # train_network(opt)

    # we write an empty *.txt file with the completed experiment
    flag_completed_dir = join(output_path, 'flag_completed')
    os.makedirs(flag_completed_dir, exist_ok=True)
    file_object = open(join(flag_completed_dir, "complete_%s.txt" % str(opt.id)), "w")
    file_object.close()