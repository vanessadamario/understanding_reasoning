import numpy as np
import json
from PIL import Image
from os.path import join


MNIST_DIGITS = [str(k_) for k_ in np.arange(10)]
MNIST_ID = np.arange(10)
RELATIONS = ["left_of", "right_of", "below", "above"]

# TODO: GENERATE VOCABULARY
vocab = {k_: int(k_) for k_ in MNIST_DIGITS}  # MNIST
for i_, r_ in enumerate(RELATIONS):
    vocab[r_] = i_ + len(MNIST_DIGITS)


def change_size(img, digit_size, output_size=28, position_x=-0, position_y=-0):
    """
    We rescale the digit and we put it to the center.
    GIVE PRIORITY TO THIS TRANSFORMATION OVER ANY OTHER.
    :param img: image as numpy array
    :param digit_size: dimension of the digit
    :param output_size: size of the output
    :param position_x: position with respect to the center, in [-output_size//2+digit_size//2, output_size//2-digit_size//2]
    :param position_y: position with respect to the center, in [-output_size//2+digit_size//2, output_size//2-digit_size//2]
    :returns output_image: the final image
    """
    image = Image.fromarray(img)
    tmp = image.resize(size=(digit_size, digit_size))
    output_image = np.zeros((output_size, output_size))
    yy, xx = np.meshgrid(np.arange(output_size//2 - digit_size//2 + position_x,
                                   output_size//2 + digit_size//2 + position_x),
                         np.arange(output_size//2 - digit_size//2 + position_y,
                                   output_size//2 + digit_size//2 + position_y))
    output_image[xx, yy] = tmp
    return output_image


def invert_dict(d):
    return {v: k for k, v in d.items()}


class Object(object):
    def __init__(self, pos=None, shape=None, digit_size=12, image_size=44):
        self.pos = pos
        self.shape = shape
        self.digit_size = digit_size
        self.image_size = image_size
        if self.pos is None:
            self.set_position()

    def set_position(self):
        posx = np.random.choice(np.arange(self.image_size - self.digit_size))
        posy = np.random.choice(np.arange(self.image_size - self.digit_size))
        self.pos = [posx, posy]

    def overlap(self, other):
        # digit_size is also min_dist
        return (abs(self.pos[0] - other.pos[0]) < self.digit_size and
                abs(self.pos[1] - other.pos[1]) < self.digit_size)

    def relate(self, rel, other):
        shift = 0
        if rel == 'left_of':
            return self.pos[0] + shift < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0] + shift
        if rel == 'above':
            return self.pos[1] > other.pos[1] + shift
        if rel == 'below':
            return self.pos[1] + shift < other.pos[1]
        raise ValueError(rel)


class DataGenerator(object):
    def __init__(self, data_path, variety, ch=3, image_size=44):
        self.variety = variety
        self.ch = ch
        self.image_size = image_size
        self.train_combinations = None
        self.test_combinations = None
        self.data_path = data_path
        self.loaded_mnist = False
        self.generate_combinations()

    def generate_combinations(self):
        train_combinations = []
        test_combinations = []
        for e_lhs in MNIST_ID:
            elements_rhs_tr = np.random.choice(np.delete(MNIST_ID, e_lhs), size=self.variety)
            elements_rhs_ts = np.delete(MNIST_ID, np.append(elements_rhs_tr, e_lhs))
            for e_rhs_tr in elements_rhs_tr:
                for r_ in RELATIONS:
                    train_combinations.append([e_lhs, vocab[r_], e_rhs_tr])
            for e_rhs_ts in elements_rhs_ts:
                for r_ in RELATIONS:
                    test_combinations.append([e_lhs, vocab[r_], e_rhs_ts])
        self.train_combinations = np.array(train_combinations)
        self.test_combinations = np.array(test_combinations)

    def load_mnist(self, split):
        self.X = np.load(join(self.data_path, "x_%s.npy" % split)) / 255
        self.y = np.load(join(self.data_path, "y_%s.npy" % split))
        self.split = split

    def _generate_img(self, shape_lhs, relation, shape_rhs, label, image_size=44):
        flag_overlap = True
        while flag_overlap:
            o_lhs = Object(shape=shape_lhs, image_size=44)
            o_rhs = Object(shape=shape_rhs, image_size=44)
            flag_overlap = o_lhs.overlap(o_rhs)
        if label != o_lhs.relate(relation, o_rhs):
            old_pose_lhs = o_lhs.pos
            o_lhs.pos = o_rhs.pos
            o_rhs.pos = old_pose_lhs

        dig_size = o_lhs.digit_size
        img_size = o_lhs.image_size
        img = np.zeros((img_size, img_size))

        idx_lhs = np.random.choice(np.argwhere(self.y == int(shape_lhs)).squeeze())
        idx_rhs = np.random.choice(np.argwhere(self.y == int(shape_rhs)).squeeze())

        yy, xx = np.meshgrid(np.arange(o_lhs.pos[0], o_lhs.pos[0] + dig_size),
                             np.arange(o_lhs.pos[1], o_lhs.pos[1] + dig_size))

        img_ = change_size(self.X[idx_lhs], digit_size=dig_size, output_size=dig_size)
        img[xx, yy] = img_

        img2_ = change_size(self.X[idx_rhs], digit_size=dig_size, output_size=dig_size)

        yy, xx = np.meshgrid(np.arange(o_rhs.pos[0], o_rhs.pos[0] + dig_size),
                             np.arange(o_rhs.pos[1], o_rhs.pos[1] + dig_size))
        img[xx, yy] = img2_
        return img

    def generate_data_matrix(self, savepath, n, split=None):
        split_list = ["train", "valid", "test"] if split is None else [split]
        if split is not None:
            if isinstance(n, int):
                n = [n]
        else:
            if not isinstance(n, list):
                raise ValueError("A list of number of examples per split is needed.")

        inv_vocab = invert_dict(vocab)

        for split_, n_ in zip(split_list, n):
            self.load_mnist(split_)

            combinations = self.train_combinations if self.split == "train"\
                else self.test_combinations
            n_combs = combinations.shape[0]

            examples_per_el = n_ // n_combs
            x_relations = np.zeros((n_combs * examples_per_el, self.ch, self.image_size, self.image_size))
            q_relations = np.zeros((n_combs * examples_per_el, 3), dtype=int)
            y_relations = np.zeros(n_combs * examples_per_el, dtype=int)

            count = 0
            for i, comb in enumerate(combinations):
                q_relations[i*examples_per_el:(i+1)*examples_per_el] = comb
                for l_ in range(examples_per_el):
                    x_relations[i*examples_per_el+l_, 1, :, :] = self._generate_img(inv_vocab[comb[0]],
                                                                                    inv_vocab[comb[1]],
                                                                                    inv_vocab[comb[2]],
                                                                                    label=count % 2)
                    y_relations[i*examples_per_el+l_] = count % 2
                    count += 1

            rnd_idx = np.arange(n_combs * examples_per_el)
            np.random.shuffle(rnd_idx)

            np.save(join(savepath, "x_%s.npy" % self.split), x_relations[rnd_idx])
            np.save(join(savepath, "y_%s.npy" % self.split), y_relations[rnd_idx])
            np.save(join(savepath, "q_%s.npy" % self.split), q_relations[rnd_idx])

        with open(join(savepath, 'vocab.json'), 'w') as outfile:
            json.dump(vocab, outfile)

        return