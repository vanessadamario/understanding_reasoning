import torch
import numpy as np
import os
from os.path import join
from torch.utils.data import Dataset, DataLoader


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
                 split_files=False
                 ):
        self.all_questions = all_questions
        self.all_labels = all_labels
        self.all_feats = all_feats
        self.split_files = split_files

    def __getitem__(self, index):
        """Generates one sample of data
        :param index: index for pick one example"""
        if self.split_files:
            input_ = np.load(join(self.all_feats, '%i.npy' % index))
        else:
            input_ = self.all_feats[index]
        feat = torch.FloatTensor(input_.astype(np.float32))
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
                 split="train"
                 ):
        """Initialization
                :param opt: the experiment class
                :param split: string with the dataset split """
        path_data = opt.dataset.dataset_id_path
        if opt.dataset.experiment_case == 0:
            questions_file = "questions_query_%s.npy" % split
            answers_file = "answers_query_%s.npy" % split
        elif opt.dataset.experiment_case == 1:
            questions_file = "questions_%s.npy" % split
            answers_file = "answers_%s.npy" % split
        elif opt.dataset.experiment_case == 2:
            questions_file = "dense_questions_%s.npy" % split
            answers_file = "dense_answers_%s.npy" % split
        else:
            raise ValueError("Experiment case %i does not exist. " % opt.dataset.experiment_case)

        if opt.dataset.experiment_case == 2:
            np_all_questions = np.load(join(path_data, questions_file))
            _, q_ = np_all_questions.shape
            all_questions = _dataset_to_tensor(np_all_questions.reshape(-1,))
            all_labels = _dataset_to_tensor(np.load(join(path_data, answers_file)).reshape(-1,))
            np_all_feats = np.load(join(path_data, "feats_%s.npy" % split))
            all_feats = np.array([im__ for i, im__ in enumerate(np_all_feats) for j in range(q_)])

        else:
            files_feats = [f_ for f_ in os.listdir(path_data) if f_.startswith('feats_%s' % split)]
            if len(files_feats) > 1:
                raise ValueError('Ambiguity in reading the feature file')
            if files_feats[0].endswith('npy'):
                split_files = False
                try:
                    all_feats = np.load(join(path_data, "feats_%s.npy" % split))
                except:
                    all_feats = np.load(join(path_data, "feats_%s.npy" % split), allow_pickle=True)
                if np.ndim(all_feats) == 3:
                    n, dim_x, dim_y = all_feats.shape
                    all_feats = all_feats.reshape(n, 1, dim_x, dim_y)
            else:
                split_files = True
                all_feats = join(path_data, 'feats_%s' % split)


            all_questions = _dataset_to_tensor(np.load(join(path_data, questions_file)))
            all_labels = _dataset_to_tensor(np.load(join(path_data, answers_file)))

        self.dataset = DataTorch(all_feats, all_questions, all_labels, split_files=split_files)
        shuffle_data = True  # if split == "train" else False
        super(DataTorchLoader, self).__init__(self.dataset,
                                              batch_size=opt.hyper_opt.batch_size,
                                              shuffle=shuffle_data)

    def __enter__(self):
        return self




