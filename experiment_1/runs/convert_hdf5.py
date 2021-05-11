import io
import os
import h5py
import numpy as np
from os.path import join


def hdf5_to_single_numpy(id, dataset_path):
    split_list = ['train', 'valid', 'test']
    for split_ in split_list:
        try:
            print(dataset_path)
            all_feats = h5py.File(join(dataset_path, 'feats_%s.hdf5' % split_), 'r')
        except:
            raise ValueError('hdf5 not found')
        output_path = join(dataset_path, 'feats_%s' % split_)
        os.makedirs(output_path, exist_ok=True)

        if split_ == 'train':
            n_files = 10000
        else:
            n_files = 200
        for k_ in range(id*n_files, (id+1)*n_files):
            img_ = all_feats['features'][k_]
            np.save(join(output_path, '%i.npy' % k_), img_)
