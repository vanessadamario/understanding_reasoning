import torch
import numpy as np
from os.path import join
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


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
                 all_labels
                 ):
        self.all_questions = all_questions
        self.all_labels = all_labels
        self.all_feats = all_feats

    def __getitem__(self, index):
        """Generates one sample of data
        :param index: index for pick one example"""
        feat = torch.FloatTensor(self.all_feats[index].astype(np.float32))
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

        all_questions = _dataset_to_tensor(np.load(join(path_data, "questions_%s.npy" % split)))
        all_labels = _dataset_to_tensor(np.load(join(path_data, "answers_%s.npy" % split)))
        all_feats = np.load(join(path_data, "feats_%s.npy" % split))
        n, dimx, dimy = all_feats.shape
        all_feats = all_feats.reshape(n, 1, dimx, dimy)

        self.dataset = DataTorch(all_feats, all_questions, all_labels)
        shuffle_data = True if split == "train" else False
        super(DataTorchLoader, self).__init__(self.dataset,
                                              batch_size=opt.hyper_opt.batch_size,
                                              shuffle=shuffle_data)

    def __enter__(self):
        return self


