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

# === Definition of modules for NMN === #

def shape_module(shape):
    return "Shape[{}]".format(shape)


def binary_shape_module(shape):
    return "Shape2[{}]".format(shape)


def color_module(color):
    return "Color[{}]".format(color)


def binary_color_module(color):
    return "Color2[{}]".format(color)


def relation_module(relation):
    return "Relate[{}]".format(relation)


def unary_relation_module(relation):
    return "Relate1[{}]".format(relation)


image_size = 64
min_obj_size = 10
max_obj_size = 15
num_shapes=len(SHAPES)
num_colors=1
rhs_variety=1
split='systematic'
num_repeats=10
num_repeats_eval=10
fontsize=15
font='./../arial.ttf'
FONT_OBJECTS = {font_size: ImageFont.truetype(font) for font_size in range(10, 16)}
vocab = SHAPES[:num_shapes]
question_words = (['<NULL>', '<START>', '<END>', 'is', 'there', 'a', 'green'] + vocab + RELATIONS)
question_vocab = {word: i for i, word in enumerate(question_words)}

# num_objects = 2  # this must be given externally

program_words = (['<NULL>', '<START>', '<END>', 'scene', 'And']
                 + [color_module('green')]
                 + [shape_module(shape) for shape in vocab]
                 + [binary_color_module('green') ]
                 + [binary_shape_module(shape) for shape in vocab]
                 + [relation_module(rel) for rel in RELATIONS]
                 + [unary_relation_module(rel) for rel in RELATIONS])
program_vocab = {word: i for i, word in enumerate(program_words)}


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


def generate_scene_(rng, sampler, objects=[], num_objects=0, restrict=False, **kwargs):
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

        new_object = get_random_spot_(rng, objects)
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


def __evaluate_activations(X, R, Y, pos_x=[5, 5], pos_y=[20, 20],
                         path_vocab_dataset='.',
                         path_model='.',
                         model_file='something.pt.best',
                         img_npy=None,
                         flip_question=False,
                         normalization=False,
                         show=True,
                         gpu_id=0):
    """ Old version of evaluate_activations """

    os.environ["CUDA_VISIBLE_DEVICES"] = "%i" % gpu_id
    device = torch.device('cuda:%i' % gpu_id if torch.cuda.is_available() else 'cpu')

    # put the flipped option
    # same image, flipped objects

    question_idx = [X, R, Y]

    uniform_dist = [1.0 / len(vocab)]*len(vocab)
    sampler_class = LongTailSampler(uniform_dist)

    test_sampler = sampler_class(True,  3, vocab)

    seed = 1
    rng = np.random.RandomState(seed)

    obj1 = Object(fontsize=10, angle=0, pos=pos_x)
    obj2 = Object(fontsize=10, angle=0, pos=pos_y)

    vocab_dataset = load_vocab(path_vocab_dataset)
    question = [vocab_dataset['question_idx_to_token'][q_] for q_ in question_idx]

    if img_npy is None:
        obj1.shape = question[0]
        obj2.shape = question[2]
        scene = generate_scene_(rng, test_sampler, objects=[obj1, obj2], restrict=True, relation=question[1])
        img = draw_scene_(scene)
        img_npy = np.array(img).transpose(2, 0, 1) / 255

    tree_model, tree_kwargs = load_execution_engine(join(path_model, model_file))

    if torch.cuda.is_available():
    # with torch.cuda.device(gpu_id):
        tree_model.cuda()
    # tree_model.to(device)
    tree_model.eval()

    if flip_question:
        question_idx = [Y, R, X]

    question_torch = torch.tensor(question_idx)
    feats_torch = torch.tensor((np.array(img_npy).reshape(1, 3, 64, 64)).astype('float32'))

    question_var = Variable(question_torch.to(device))
    feats_var = Variable(feats_torch.to(device))

    # question_var = question_var.to(device)
    # feats_var = feats_var.to(device)
    # if torch.cuda.device(gpu_id):
    #    question_var = Variable(question_torch.cuda())
    #    feats_var = Variable(feats_torch.cuda())

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
    print('res shape', res.shape)
    if normalization:
        normalization_const = np.load(join(path_model,
                                           '%s_norm_const.npy' % model_file.split('.')[0]))
    else:
        # we remove the sentinel and the image
        normalization_const = np.ones((res.shape[1]-2, res.shape[2]))

    maps = res.cpu().detach().numpy()[0]  # one sample at the time
    print('maps shape', maps.shape)
    for i_ in range(2, maps.shape[0]):
        for j_ in range(maps.shape[1]):
            maps[i_, j_, :, :] = maps[i_, j_, :, :]/normalization_const[i_-2, j_]

    if show:
        for id_map in range(maps.shape[1]):
            print(id_map)
            fig, ax = plt.subplots(figsize=(15, 5), ncols=maps.shape[0])
            for id_module in range(maps.shape[0]):
                im = ax[id_module].imshow(maps[id_module, id_map])
                plt.colorbar(im, ax=ax[id_module], fraction=0.046)
                ax[id_module].set_axis_off()
            plt.show()

    tree_scores = tree_model.classifier(res[:, -1, :, :, :])
    pred = F.softmax(tree_scores, dim=1).cpu().detach().numpy()

    del tree_model, question_torch, feats_torch, question_var, feats_var
    del question_embed, stem_image, res, tree_scores

    return img_npy, maps, pred


def generate_scene_dataset(path_vocab_dataset,
                           obj_x='P',
                           obj_y='Y',
                           rel='above',
                           obj_flip='Z',
                           flip_question=True,
                           img_size=64):
    """ We generate a dataset containing images and questions,
    where we move the two objects around, with the option of
    flipping the question and flipping the objects.
    :param path_vocab_dataset: str, path to the vocab.json file
    :param obj_x: str, first object in the question
    :param obj_y: str, second object in the question
    :param rel: str, relation (above, below, left of, right of)
    :param obj_flip: str, if we want to change the second object, if not put None
    :param flip_question: bool, same images, flipped objects in the question
    :param img_size: int, size of the square image dimension
    :returns img: numpy array of dimensions (#data_size, channels, image_size, image_size)
    :returns qst: numpy array of dimensions (#data_size, 3)
    """

    flip_lst = [False, True] if flip_question else [False]
    tuple_lst = []
    array_positions = np.arange(5, 65, 20)
    vocab_data = load_vocab(path_vocab_dataset)
    question_id = [vocab_data['question_token_to_idx'][q_] for q_ in [obj_x, obj_y]]
    tuple_lst.append((question_id[0], question_id[1]))
    if obj_flip:
        question_id = [vocab_data['question_token_to_idx'][q_] for q_ in [obj_x, obj_flip]]
        tuple_lst.append((question_id[0], question_id[1]))
    rel_idx = vocab_data['question_token_to_idx'][rel]

    img = np.zeros((len(flip_lst),
                    len(tuple_lst),
                    array_positions.size ** 2,
                    array_positions.size ** 2,
                    3,  # RGB channels
                    img_size,
                    img_size),
                   dtype=np.float32)
    qst = np.zeros((len(flip_lst),
                    len(tuple_lst),
                    array_positions.size ** 2,
                    array_positions.size ** 2,
                    3),
                   dtype=int)  # elements in the question

    uniform_dist = [1.0 / len(vocab)] * len(vocab)
    sampler_class = LongTailSampler(uniform_dist)
    test_sampler = sampler_class(True, 3, vocab)

    seed = 1
    rng = np.random.RandomState(seed)

    for id_flip, flip in enumerate(flip_lst):  # flip or not the question, not the image
        for id_tuple, XX in enumerate(tuple_lst):
            obj1 = Object(fontsize=10,
                          angle=0,
                          pos=[0, 0],
                          shape=vocab_data['question_idx_to_token'][XX[0]])
            obj2 = Object(fontsize=10,
                          angle=0,
                          pos=[0, 0],
                          shape=vocab_data['question_idx_to_token'][XX[1]])
            tmp_qst = [XX[1], rel_idx, XX[0]] if flip else [XX[0], rel_idx, XX[1]]

            for id_Y_x, Y_x in enumerate(array_positions):
                for id_Y_y, Y_y in enumerate(array_positions):
                    obj2.pos = [Y_x, Y_y]
                    for id_X_x, X_x in enumerate(array_positions):
                        for id_X_y, X_y in enumerate(array_positions):
                            obj1.pos = [X_x, X_y]
                            scene = generate_scene_(rng, test_sampler,
                                                    objects=[obj1, obj2],
                                                    restrict=True,
                                                    relation=rel)
                            img_ = draw_scene_(scene)
                            img[id_flip, id_tuple,
                                array_positions.size * id_Y_x + id_Y_y,
                                array_positions.size * id_X_x + id_X_y] = np.array(img_).transpose(2, 0, 1) / 255.

                            qst[id_flip,
                                id_tuple,
                                array_positions.size * id_Y_x + id_Y_y,
                                array_positions.size * id_X_x + id_X_y] = tmp_qst

    return img, qst


def evaluate_activations(X, R, Y, pos_x=[5, 5], pos_y=[20, 20],
                         images=None,
                         questions=None,
                         path_vocab_dataset='.',
                         path_model='.',
                         model_file='something.pt.best',
                         img_npy=None,
                         flip_question=False,
                         normalization=False,
                         show=False,
                         gpu_id=0):
    """ Evaluate activations maps across the three modules.
    It can be use for single points as well as for larger sets,
    as long as they fit in a single GPU.
    :param X: int, index for the first object in the question
    :param R: int, index for the relation
    :param Y: int, index for the second object in the question
    :param pos_x: list, contains ints for first object's x and y coordinates
    :param pos_y: list, contains ints for second object's x and y coordinates
    :param path_vocab_dataset: str, path to the vocab.json file
    :param path_model: str, path to the folder containing models
    :param model_file: str, name of the model
    :param img_npy: default None, otherwise to be used with X,R,Y on single example
    :param flip_question: bool, default False
    :param normalization: bool, normalization based on the
        maximum activity of each filter (64 per module)
    :param show: default False, shows the activity map for a single data point
    :param gpu_id: int, default 0, index of the gpu to be used

    :return img_npy: numpy array containing the scene
    :return maps: numpy array containing the activations
    :return pred: numpy array containing the prediction for each scene
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = "%i" % gpu_id
    device = torch.device('cuda:%i' % gpu_id if torch.cuda.is_available() else 'cpu')

    if (images is not None) and (questions is not None):  # we are given a data set
        data_shape = [dim_ for dim_ in questions.shape[:-1]]  # batch size
        img_size = images.shape[-1]
        img_npy = images.reshape(-1, 3, img_size, img_size)
        question_idx = questions.reshape(-1, 3)
    else:  # if not we generate it
        img_size = 64
        data_shape = [1]
        question_idx = [X, R, Y]
        uniform_dist = [1.0 / len(vocab)]*len(vocab)
        sampler_class = LongTailSampler(uniform_dist)
        test_sampler = sampler_class(True,  3, vocab)
        seed = 1
        rng = np.random.RandomState(seed)

        obj1 = Object(fontsize=10, angle=0, pos=pos_x)
        obj2 = Object(fontsize=10, angle=0, pos=pos_y)
        vocab_dataset = load_vocab(path_vocab_dataset)
        question = [vocab_dataset['question_idx_to_token'][q_] for q_ in question_idx]

        if img_npy is None:
            obj1.shape = question[0]
            obj2.shape = question[2]
            scene = generate_scene_(rng, test_sampler, objects=[obj1, obj2], restrict=True, relation=question[1])
            img = draw_scene_(scene)
            img_npy = ((np.array(img).transpose(2, 0, 1) / 255).reshape(1, 3, img_size, img_size)).astype('float32')

        if flip_question:
            question_idx = [Y, R, X]

    tree_model, tree_kwargs = load_execution_engine(join(path_model, model_file))
    if torch.cuda.is_available():
        tree_model.cuda()
    tree_model.eval()

    question_torch = torch.tensor(question_idx)
    feats_torch = torch.tensor(img_npy)

    question_var = Variable(question_torch.to(device))
    feats_var = Variable(feats_torch.to(device))

    question_embed = tree_model.question_embeddings(question_var)
    stem_image = tree_model.stem(feats_var)

    res = _shnmn_func(question=question_embed,
                      img=stem_image.unsqueeze(1),
                      num_modules=tree_model.num_modules,
                      alpha=tree_model.alpha,
                      tau_0=Variable(tree_model.tau_0),
                      tau_1=Variable(tree_model.tau_1),
                      func=tree_model.func)
    print('res shape', res.shape)

    del question_embed, stem_image, question_var, feats_var, question_torch, feats_torch

    tree_scores = tree_model.classifier(res[:, -1, :, :, :])
    del tree_model

    pred = F.softmax(tree_scores, dim=1).cpu().detach().numpy()
    maps = res.cpu().detach().numpy()  # [0]  # one sample at the time
    del tree_scores, res

    if normalization:
        normalization_const = np.load(join(path_model,
                                           '%s_norm_const.npy' % model_file.split('.')[0]))
    else:
        normalization_const = np.ones((maps.shape[1]-2, maps.shape[2]))

    print('maps shape', maps.shape)
    for i_ in range(2, maps.shape[1]):
        for j_ in range(maps.shape[2]):
            maps[:, i_, j_, :, :] = maps[:, i_, j_, :, :]/normalization_const[i_-2, j_]

    data_shape_ = data_shape + [dim_ for dim_ in maps.shape[1:]]

    if show:
        maps_ = maps[0]
        for id_map in range(maps_.shape[1]):
            print(id_map)
            fig, ax = plt.subplots(figsize=(15, 5), ncols=maps.shape[0])
            for id_module in range(maps_.shape[0]):
                im = ax[id_module].imshow(maps_[id_module, id_map])
                plt.colorbar(im, ax=ax[id_module], fraction=0.046)
                ax[id_module].set_axis_off()
            plt.show()

    img_npy = img_npy.reshape(data_shape + [3, img_size, img_size])
    maps = maps.reshape(data_shape_)  # reshape the activation maps
    pred = pred.reshape(data_shape + [2])  # given that we have two classes
    return img_npy, maps, pred


def get_random_spot_(rng, objects, rel=None,  rel_holds=False, rel_obj=0):
    """Get a spot for a new object that does not overlap with existing ones."""
    # then, select the object size
    size = rng.randint(min_obj_size, max_obj_size + 1)
    angle = 0
    obj = Object(size, angle)

    min_center = obj.rotated_size // 2 + 1
    max_center = image_size - obj.rotated_size // 2 - 1

    if rel is not None:
        if rel_holds == False:
            # do not want the relation to be true
            max_center_x = objects[rel_obj].pos[0] if rel == 'left_of' else max_center
            min_center_x = objects[rel_obj].pos[0] if rel == 'right_of' else min_center
            max_center_y = objects[rel_obj].pos[1] if rel == 'below' else max_center
            min_center_y = objects[rel_obj].pos[1] if rel == 'above' else min_center
        else:
            # want the relation to be true
            min_center_x = objects[rel_obj].pos[0] if rel == 'left_of' else min_center
            max_center_x = objects[rel_obj].pos[0] if rel == 'right_of' else max_center
            min_center_y = objects[rel_obj].pos[1] if rel == 'below' else min_center
            max_center_y = objects[rel_obj].pos[1] if rel == 'above' else max_center

        if min_center_x >= max_center_x: return None
        if min_center_y >= max_center_y: return None

    else:
        min_center_x = min_center_y = min_center
        max_center_x = max_center_y = max_center

    for attempt in range(10):
        x = rng.randint(min_center_x, max_center_x)
        y = rng.randint(min_center_y, max_center_y)
        obj.pos = (x, y)

        # make sure there is no overlap between bounding squares
        if (any([abs(obj.pos[0] - other.pos[0]) < 5 for other in objects]) or
            any([abs(obj.pos[1] - other.pos[1]) < 5 for other in objects])):
            continue
        if any([obj.overlap(other) for other in objects]):
            continue
        return obj
    else:
        return None


def generate_image_and_question_(pair, sampler, rng, label, rel, num_objects):
    # x rel y has value label where pair == (x, y)
    x, y = pair

    if label:
        obj1 = get_random_spot_(rng, [])
        obj2 = get_random_spot_(rng, [obj1])
        if not obj2 or not obj1.relate(rel, obj2): return None, None, None, False, 'a'
        obj1.shape = x
        obj2.shape = y
        scene = generate_scene_(rng, sampler, objects=[obj1, obj2], num_objects=num_objects, restrict=False, relation=rel)
    else:
        # first generate a scene
        obj1 = get_random_spot_(rng, [])
        obj2 = get_random_spot_(rng, [obj1], rel=rel, rel_holds=False)
        if not obj2 or obj1.relate(rel, obj2): return None, None, None, False, 'b'
        obj1.shape = x
        obj2.shape = y

        scene = generate_scene_(rng, sampler, objects=[obj1, obj2], num_objects=num_objects, restrict=True, relation=rel)
        # choose x,y,x', y' st. x r' y, x r y', x' r y holds true

        if num_objects >=4:

            obj3 = scene[2]  # x'
            obj4 = scene[3]  # y'

            if not obj1.relate(rel, obj4): return None, None, None, False, 'c'
            elif not obj3.relate(rel, obj2): return None, None, None, False, 'd'

    color1 = "green"
    color2 = "green"
    shape1 = x
    shape2 = y
    question = [x, rel, y]
    program = ["<START>", relation_module(rel),
               shape_module(shape1), "scene",
               shape_module(shape2), "scene",
               "<END>"]

    return scene, question, program, True, 'f'


def gen_data_(obj_pairs, sampler, seed, prefix, num_objects, vocab_dataset,
              output_path='.'):
    """ Data generation process. Given the data and the object pairs we create a dataset.
    :param obj_pairs:
    :param sampler:
    :param seed:
    :param prefix:
    :param num_objects:
    :param vocab_dataset:
    :param output_path:
    :return:
    """
    num_examples = len(obj_pairs)

    max_question_len = 3
    max_program_len = 7
    # list containing numpy arrays of 3 elements
    presampled_relations_idx = list(obj_pairs[:, 1])
    presampled_relations = [vocab_dataset['question_idx_to_token'][i_] for i_ in presampled_relations_idx]

    obj_pairs_ = obj_pairs[:, ::2]
    obj_pairs = [[vocab_dataset['question_idx_to_token'][ob[0]],
                  vocab_dataset['question_idx_to_token'][ob[1]]]
                    for ob in obj_pairs_]

    with h5py.File(join(output_path, prefix + '_questions.h5'), 'w') as dst_questions, h5py.File(join(output_path, prefix + '_features.h5'), 'w') as dst_features:
        features_dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
        features_dataset = dst_features.create_dataset('features', (num_examples,), dtype=features_dtype)
        questions_dataset = dst_questions.create_dataset('questions', (num_examples, max_question_len), dtype=np.int64)
        programs_dataset = dst_questions.create_dataset('programs', (num_examples, max_program_len), dtype=np.int64)
        answers_dataset = dst_questions.create_dataset('answers', (num_examples,), dtype=np.int64)
        image_idxs_dataset = dst_questions.create_dataset('image_idxs', (num_examples,), dtype=np.int64)

        i = 0
        rejection_sampling = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0}

        # different seeds for train/dev/test
        rng = np.random.RandomState(seed)
        before = time.time()
        scenes = []
        while i < len(obj_pairs):
            scene, question, program, success, key = generate_image_and_question_(
                obj_pairs[i], sampler, rng, (i % 2) == 0, presampled_relations[i], num_objects)
            rejection_sampling[key] += 1
            if success:
                scenes.append(scene)
                buffer_ = io.BytesIO()
                image = draw_scene_(scene)
                image.save(buffer_, format='png')
                buffer_.seek(0)
                features_dataset[i] = np.frombuffer(buffer_.read(), dtype='uint8')
                questions_dataset[i] = [question_vocab[w] for w in question]
                programs_dataset[i] = [program_vocab[w] for w in program]
                answers_dataset[i] = int((i % 2) == 0)
                image_idxs_dataset[i] = i

                i += 1
                if i % 1000 == 0:
                    time_data = "{} seconds per example".format((time.time() - before) / i )
                    print(time_data)
                print("\r>> Done with %d/%d examples : %s " %(i+1, len(obj_pairs),  rejection_sampling), end = '')
                sys.stdout.flush()

    print("{} seconds per example".format((time.time() - before) / len(obj_pairs) ))

    with open(join(output_path, prefix + '_scenes.json'), 'w') as dst:
        json.dump(scenes, dst, indent=2, cls=CustomJSONEncoder)


def generate_in_sample(path_data, n_objects=2, in_sample_test_dim=6000, output_path='.'):
    vocab_dataset = load_vocab(join(path_data, 'vocab.json'))
    question_tr = h5py.File(join(path_data, 'train_questions.h5'), 'r')

    unique_questions = np.unique(np.array(question_tr['questions']), axis=0)
    rep = in_sample_test_dim // unique_questions.shape[0]
    new_questions_ = np.zeros((rep * unique_questions.shape[0], 3), dtype=int)
    for id_q_, q_ in enumerate(unique_questions):
        new_questions_[id_q_ * rep:(id_q_ + 1) * rep, :] = q_ * np.ones((rep, 3), dtype=int)

    seed = 1
    uniform_dist = [1.0 / len(vocab)] * len(vocab)
    sampler_class = LongTailSampler(uniform_dist)
    train_sampler = sampler_class(False, 1, vocab)
    gen_data_(new_questions_,
              sampler=train_sampler,
              seed=seed,
              prefix='test_in_sample',
              num_objects=n_objects,
              vocab_dataset=vocab_dataset,
              output_path=output_path)

    return


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


def read_output_files(slurm_id_lst,
                      path_output,
                      output_starts_with='output_'):
    """
    ----------------
    Parameters
        slurm_id_lst: str, slurm id from which we extract the hyperparameters
        path_model: str, root path to the files
    ----------------
    Returns
        test error: np.array of test error
    """
    if slurm_id_lst is None:
        output_list_files = path_output
    else:
        output_list_files = [join(path_output, '%s%s.pt.best.h5' % (output_starts_with, slurm_id)) for slurm_id in slurm_id_lst]

    test_accuracy = []

    for output_file in output_list_files:
        output = h5py.File(output_file, 'r')
        correct = np.array(list(output['correct']))

        test_accuracy.append(np.sum(correct).astype('float') / correct.size)
        probs = np.array(list(output['probs']))
        scores = np.array(list(output['scores']))

    test_accuracy = np.array(test_accuracy)
    return 1 - test_accuracy

