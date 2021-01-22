import os
import json


def check_update(output_path):
    """ This function checks if the experiments have already been performed
    and writes everything in a json file. Once each experiment has been
    completed, the program write an empty txt file called complete_<EXP_N>.txt.
    :param output_path: the output path for the *.json file, and for the
    *.txt files that tell us if the experiments were completed
    . necessary to change the *.json
    """

    flag_txt = [(f_.split('_')[-1]).split('.')[0]
                for f_ in os.listdir(output_path + 'flag_completed')]
    # this is a list of string

    # here we open the json and we change the flag
    with open(output_path + 'train.json') as infile:
        info = json.load(infile)

    # set all to False
    for exp_id in range(len(info)):
        info[str(exp_id)]['train_completed'] = False

    for exp_id in flag_txt:
        info[exp_id]['train_completed'] = True

    with open(output_path + 'train.json', 'w') as outfile:
        json.dump(info, outfile)