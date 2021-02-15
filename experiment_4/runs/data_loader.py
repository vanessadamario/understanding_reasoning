import io
import os
import h5py
from PIL import Image
import torch
import numpy as np
from os.path import join
from torch.utils.data import Dataset, DataLoader

PATH_SQOOP_DATASET = "/om/user/vanessad/understanding_reasoning/experiment_4/data_generation/sysgen_sqoop/sqoop_variety_1"


def _dataset_to_tensor(dset, mask=None, dtype=None):
    arr = np.asarray(dset, dtype=np.int64 if dtype is None else dtype)
    if mask is not None:
        arr = arr[mask]
    tensor = torch.LongTensor(arr)
    return tensor


class DataTorch(Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self,
                 all_feats,
                 all_questions,
                 all_labels,
                 convertPIL=False,
                 h5_file=False):
        self.all_questions = all_questions
        self.all_labels = all_labels
        self.all_feats = all_feats
        if h5_file:
            self.all_feats = all_feats['features']
        self.convertPIL = convertPIL
        self.h5_file = h5_file

    def __getitem__(self, index):
        """Generates one sample of data
        :param index: index for pick one example"""
        input_ = self.all_feats[index]
        if not self.convertPIL:
            feat = torch.FloatTensor(input_.astype(np.float32))
        else:
            # print("convertPIL is true")
            feat = torch.FloatTensor(np.array(Image.open(io.BytesIO(input_))).transpose(2, 0, 1) / 255.0)
            # print(feat.shape)
        question = self.all_questions[index]
        label = self.all_labels[index]

        return feat, question, label

    def __len__(self):
        """Denotes the total number of samples"""
        return self.all_questions.size(0)


class DataTorchLoader(DataLoader):
    """ DataLoader from pytorch.
    It generates automatically batches """
    def __init__(self,
                 opt,
                 split="train",
                 ):
        """Initialization
                :param opt: the experiment class
                :param split: string with the dataset split """
        path_data = opt.dataset.dataset_id_path
        questions_file = "questions_%s.npy" % split
        answers_file = "answers_%s.npy" % split

        all_questions = _dataset_to_tensor(np.load(join(path_data, questions_file)))
        all_labels = _dataset_to_tensor(np.load(join(path_data, answers_file)))

        files_feats = [f_ for f_ in os.listdir(path_data) if f_.startswith('feats_%s' % split)]
        if len(files_feats) > 1:
            raise ValueError('Ambiguity in reading the feature file')
        ext = files_feats[0].split('.')[-1]
        if ext == 'npy':
            h5_file = False
            try:
                all_feats = np.load(join(path_data, "feats_%s.npy" % split))
            except:
                all_feats = np.load(join(path_data, "feats_%s.npy" % split), allow_pickle=True)
            if np.ndim(all_feats) == 3:
                n, dim_x, dim_y = all_feats.shape
                all_feats = all_feats.reshape(n, 1, dim_x, dim_y)
        elif ext == 'hdf5':
            h5_file = True
            all_feats = h5py.File(join(path_data, 'feats_%s.hdf5' % split), 'r')
        else:
            raise ValueError('File extension not recognized')

        convertPIL = opt.dataset.dataset_id_path == PATH_SQOOP_DATASET
        # TODO: change here, pass the argument
        self.dataset = DataTorch(all_feats, all_questions, all_labels, convertPIL=convertPIL, h5_file=h5_file)
        # shuffle_data = True if split == "train" else False
        shuffle_data = True
        super(DataTorchLoader, self).__init__(self.dataset,
                                              batch_size=opt.hyper_opt.batch_size,
                                              shuffle=shuffle_data)

    def __enter__(self):
        return self




