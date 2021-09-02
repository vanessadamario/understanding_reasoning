import os
import json
from os.path import join


def modify_path(path_file,
                new_ckpt_path=None,
                new_data_path=None):
    info = json.load(open(path_file, 'r'))
    info_copy = info.copy()
    info_copy['args']['checkpoint_path'] = new_ckpt_path
    info_copy['args']['data_dir'] = new_data_path
    info_copy['args']['vocab_json'] = join(new_data_path, 'vocab.json')
    dir_path_file = os.path.dirname(path_file)
    name_file = 'dgx_' + path_file.split('/')[-1]
    output_file = join(dir_path_file, name_file)
    # print(output_file)
    print(info_copy['args']['checkpoint_path'])
    with open(output_file, 'w') as f:
        json.dump(info_copy, f)


def main():
    root_path = '/om2/user/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT_NeurIPS_revision'
    new_data_path = '/raid/poggio/home/vanessad/understanding_reasoning/CLOSURE-master/dataset_visual_bias'
    new_ckpt_path = '/raid/poggio/home/vanessad/understanding_reasoning/CLOSURE-master/results/CoGenT_NeurIPS_revision'
    for index in range(5):
        model_name = 'vector_sep_stem_%i' % index
        filename = join(root_path, model_name + '.json')
        modify_path(path_file=filename,
                    new_ckpt_path=join(new_ckpt_path, model_name),
                    new_data_path=new_data_path)


if __name__ == '__main__':
    main()
