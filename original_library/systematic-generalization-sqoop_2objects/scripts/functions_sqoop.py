import argparse
import collections
import logging
import math
import string
import time
import random
import sys
from functools import partial

import h5py
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
from os.path import join
import io
import os
import json
import torch
import h5py
import numpy as np
import PIL
from torch import nn
from torch.utils.tensorboard import SummaryWriter, FileWriter
from torch.autograd import Variable

from vr.utils import load_execution_engine, load_program_generator
from train_model import get_execution_engine
from vr.utils import load_vocab
from vr.data import ClevrDataset, ClevrDataLoader
from train_model import check_accuracy
from vr.models.shnmn import _shnmn_func
from torch.autograd import Variable
import torch.nn.functional as F


PATH_DATASET_AT_TRAIN = '/om/user/vanessad/om/user/vanessad/compositionality'

logger = logging.getLogger(__name__)
RELATIONS = ['left_of', 'right_of', 'above', 'below']
COLORS = ['red', 'green', 'blue', 'yellow', 'cyan',
          'purple', 'brown', 'gray']
SHAPES = list(string.ascii_uppercase) + ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

image_size = 64
min_obj_size = 10
max_obj_size = 15
num_shapes=len(SHAPES)
num_colors=1
num_objects=2
rhs_variety=1
split='systematic'
num_repeats=10
num_repeats_eval=10
fontsize=15
font='./../arial.ttf'
vocab = SHAPES[:num_shapes]
FONT_OBJECTS = {font_size: ImageFont.truetype(font) for font_size in range(10, 16)}

question_words = (['<NULL>', '<START>', '<END>', 'is', 'there', 'a', 'green'] + vocab + RELATIONS)
question_vocab = {word: i for i, word in enumerate(question_words)}


class Object(object):
    def __init__(self, fontsize, angle=0, pos=None, shape=None):
        self.font = FONT_OBJECTS[fontsize]
        width, self.size = self.font.getsize('A')
        self.angle = angle
        angle_rad = angle / 180 * math.pi
        self.rotated_size =  math.ceil(self.size * (abs(math.sin(angle_rad)) + abs(math.cos(angle_rad))))
        self.pos = pos
        self.shape = shape

    def overlap(self, other):
        min_dist = (self.rotated_size + other.rotated_size) // 2 + 1
        return (abs(self.pos[0] - other.pos[0]) < min_dist and
                abs(self.pos[1] - other.pos[1]) < min_dist)

    def relate(self, rel, other):
        if rel == 'left_of':
            return self.pos[0] < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0]
        if rel == 'above':
            return self.pos[1] > other.pos[1]
        if rel == 'below':
            return self.pos[1] < other.pos[1]
        raise ValueError(rel)

    def draw(self):
        img = Image.new('RGBA', (self.size, self.size))
        draw = ImageDraw.Draw(img)
        draw.text((0,0), self.shape, font=self.font, fill='green')
        return img


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Object):
            return {'size': obj.size,
                    'rotated_size': obj.rotated_size,
                    'angle': obj.angle,
                    'pos': obj.pos,
                    'shape': obj.shape
                    }
        else:
            return super().default(obj)


class Sampler:
    def __init__(self, test, seed, objects):
        self._test = test
        self._rng = np.random.RandomState(seed)
        self.objects = objects

    def _choose(self, list_like):
        return list_like[self._rng.randint(len(list_like))]

    def _rejection_sample(self, restricted=[]):
        while True:
            rand_object = self._rng.choice(self.objects)
            if rand_object not in restricted:
                return rand_object

    def sample_relation(self, *args, **kwargs):
        return self._choose(RELATIONS)

    def sample_object(self, *args, **kwargs):
        print(args)
        if len(args) > 0:
            return self._rejection_sample(args[0])
        else:
            return self._rejection_sample()


class _LongTailSampler(Sampler):
    def __init__(self, dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_probs = dist

    def sample_object(self, restricted=[], *args, **kwargs):
        if self._test:
            return self._rejection_sample(restricted=restricted)
        else:
            return self._rejection_sample(self.object_probs, restricted=restricted)

    def _rejection_sample(self, shape_probs=None, restricted=[]):
        while True:
            rand_object = self._rng.choice(self.objects, p=shape_probs)
            if rand_object not in restricted:
                return rand_object


def LongTailSampler(long_tail_dist):
    return partial(_LongTailSampler, long_tail_dist)


def generate_scene_(rng, sampler, objects=[], restrict=False, **kwargs):
    orig_objects = objects

    objects = list(orig_objects)
    place_failures = 0

    if restrict:
        restricted_obj = [obj.shape for obj in orig_objects]
    else:
        restricted_obj = []

    while len(objects) < num_objects:
        # first, select which object to draw by rejection sampling
        shape = sampler.sample_object(restricted_obj, [], **kwargs)

        new_object = get_random_spot(rng, objects)
        if new_object is None:
            place_failures += 1
            if place_failures == 10:
                # reset generation
                objects = list(orig_objects)
                place_failures = 0
            continue

        new_object.shape = shape
        objects.append(new_object)

    return objects


def draw_scene_(objects):
    img = Image.new('RGB', (image_size, image_size))
    for obj in objects:
        obj_img = obj.draw()
        obj_pos = (obj.pos[0] - obj_img.size[0] // 2,
                   obj.pos[1] - obj_img.size[1] // 2)
        img.paste(obj_img, obj_pos, obj_img)

    return img


def evaluate_activations(X, R, Y, pos_x=[5,5], pos_y=[20,20],
                         path_vocab_dataset='.',
                         path_model='.',
                         model_file='something.pt.best',
                         img_npy=None):

    question_idx = [X, R, Y]

    uniform_dist = [1.0 / len(vocab) ]*len(vocab)
    sampler_class = LongTailSampler(uniform_dist)

    test_sampler  = sampler_class(True,  3, vocab)

    seed = 1
    rng = np.random.RandomState(seed)

    obj1=Object(fontsize=10, angle=0, pos=pos_x)
    obj2=Object(fontsize=10, angle=0, pos=pos_y)

    vocab_dataset = load_vocab(path_vocab_dataset)
    question = [vocab_dataset['question_idx_to_token'][q_] for q_ in question_idx]

    if img_npy is None:
        obj1.shape = question[0]
        obj2.shape = question[2]
        scene = generate_scene_(rng, test_sampler, objects=[obj1, obj2], restrict=True, relation=question[1])
        img = draw_scene_(scene)
        img_npy = np.array(img).transpose(2, 0, 1) / 255

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tree_model, tree_kwargs = load_execution_engine(join(path_model, model_file))

    if torch.cuda.is_available():
        tree_model.cuda()
    tree_model.eval()

    question_torch = torch.tensor(question_idx)
    feats_torch = torch.tensor((np.array(img_npy).reshape(1, 3, 64, 64)).astype('float32'))

    question_var = Variable(question_torch.to(device))
    feats_var = Variable(feats_torch.to(device))

    question_embed = tree_model.question_embeddings(question_var)
    stem_image = tree_model.stem(feats_var)

    print(question_embed.shape, stem_image.shape)

    res = _shnmn_func(question=question_embed,
                      img=stem_image.unsqueeze(1),
                      num_modules=tree_model.num_modules,
                      alpha=tree_model.alpha,
                      tau_0=Variable(tree_model.tau_0),
                      tau_1=Variable(tree_model.tau_1),
                      func=tree_model.func)

    maps = res.cpu().detach().numpy()[0]

    for id_map in range(maps.shape[1]):
        print(id_map)
        fig, ax = plt.subplots(figsize=(15, 5), ncols=maps.shape[0])
        for id_module in range(maps.shape[0]):
            im = ax[id_module].imshow(maps[id_module, id_map])
            plt.colorbar(im, ax=ax[id_module], fraction=0.046)
            ax[id_module].set_axis_off()
        plt.show()

    tree_scores = tree_model.classifier(res[:, -1, :, :, :])

    return img_npy, maps, F.softmax(tree_scores, dim=1)


def retrieve_id_repeated_experiments(slurm_id=None, path_model='.', args=None):
    """ We can pass the slurm_id, or alternatively the dictionary defining the experiments.
    ----------------
    Parameters
        slurm_id: str, slurm id from which we extract the hyperparameters
        path_model: str, root path to the files
        args: dict, containing the keys for the experiment
    ----------------
    Returns
        lst_equal_files: list, list of slurm id
    """

    json_files = [f_ for f_ in os.listdir(path_model) if f_.endswith('.json')]
    lst_equal_files = []

    if args is None:
        model_json = json.load(open(join(path_model, slurm_id + '.pt.json'), 'rb'))

        dct = model_json['args']
        del dct['checkpoint_path']

        for json_f in json_files:  # list of candidate slurmids.pt.json
            try:
                loaded_file = json.load(open(join(path_model, json_f), 'rb'))
                equal = True
                dct_comparison = loaded_file['args']
                del dct_comparison['checkpoint_path']

                if len(dct_comparison) == len(dct):
                    for (k_, i_) in dct.items():
                        if i_ != dct_comparison[k_]:
                            equal = False
                            break  # we pass to the new json
                    if equal:
                        lst_equal_files.append(json_f)
                else:
                    continue

            except:
                continue
        return [f_.split('.')[0] for f_ in lst_equal_files]

    else:
        lst_equal_files = []

        for json_f in json_files:  # list of candidate slurmids.pt.json
            try:
                loaded_args = json.load(open(join(path_model, json_f), 'rb'))['args']
                equal = True
                for (key_, item_) in args.items():
                    if loaded_args[key_] != item_:
                        equal = False
                        break
                if equal:
                    lst_equal_files.append(json_f)
            except:
                continue

        return lst_equal_files
