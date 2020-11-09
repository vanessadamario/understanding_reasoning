from PIL import Image
import numpy as np
from tensorflow.keras.datasets import mnist


color_dict = {'red': [1, 0, 0],
              'green': [0, 1, 0],
              'blue': [0, 0, 1],
              'yellow': [1, 1, 0],
              'purple': [1, 0, 1]}
resize_dict = {'small': 14,
               'medium': 20,
               'large': 28}
brightness_dict = {'dark': 0.4,
                   'half': 0.7,
                   'light': 1.}


def change_size(img, digit_size, output_size=28, position_x=-0, position_y=-0):
    """ We rescale the digit and we put it to the center.
    GIVE PRIORITY TO THIS TRANSFORMATION OVER ANY OTHER.
    :param img: image as numpy array
    :param digit_size: dimension of the digit
    :param output_size: size of the output
    :param position_x: position with respect to the center, in [-output_size//2+digit_size//2, output_size//2-digit_size//2]
    :param position_y: position with respect to the center, in [-output_size//2+digit_size//2, output_size//2-digit_size//2]
    :returns output_image: the final image"""
    image = Image.fromarray(img)
    tmp = image.resize(size=(digit_size, digit_size))
    output_image = np.zeros((output_size, output_size))
    yy, xx = np.meshgrid(np.arange(output_size//2 - digit_size//2 + position_x,
                                   output_size//2 + digit_size//2 + position_x),
                         np.arange(output_size//2 - digit_size//2 + position_y,
                                   output_size//2 + digit_size//2 + position_y))
    output_image[xx, yy] = tmp
    return output_image


def contournize(img):
    """ Apply contours to the image
    :param img: image as a 2D numpy array
    :return output_image: contours
    """
    img_size = img.shape[1]
    bm_ = img != 0
    left_bm = np.hstack((np.zeros((img_size, 1)), bm_[:, :-1].astype(int)))
    right_bm = np.hstack((bm_[:, 1:].astype(int), np.zeros((img_size, 1))))
    top_bm = np.vstack((np.zeros((1, img_size)), bm_[:-1, :].astype(int)))
    bot_bm = np.vstack((bm_[1:, :].astype(int), np.zeros((1, img_size))))

    sum_shift = np.sum(np.array([left_bm, right_bm, top_bm, bot_bm]), axis=0)
    contour_bm_1 = sum_shift > 0
    contour_bm_2 = sum_shift < 4
    output_image = (contour_bm_1 * contour_bm_2 * bm_).astype(int)
    return output_image


def apply_color_brightness(img, output_color=None, bright_level=1):
    """ We color the digit and fix the brightness for the objects.
     :param img: image as numpy array, 2D array
     :param output_color: string, output color
     :param bright_level: float, value in (0, 1]
     :returns output_image: 3D numpy array """
    if output_color is None:
        return img * bright_level
    output_image = np.zeros((3, img.shape[0], img.shape[1]))
    for id_ch_, ch_ in enumerate(color_dict[output_color]):
        output_image[id_ch_, :, :] = ch_ * img * bright_level
    return output_image


def transform(img, reshape, color, bright='light', contour=False):
    # TODO: ADJUST TO THE POSSIBILITY OF HAVING A LIST OF POSSIBLE VALUES FOR FUNCTION ARGUMENT
    """ Transform the image based on the parameters we pass. In order
    Upscale
    Contour
    Colors & Brightness
    :param img: input image
    :param reshape: string, image new size: small, medium, large
    :param color: digit color
    :param bright: string for brightness_dict
    :param contour: bool value
    """
    img_ = change_size(img, resize_dict[reshape])
    if contour:
        img_ = contournize(img_)
    img_ = apply_color_brightness(img_,
                                  output_color=color,
                                  bright_level=brightness_dict[bright])
    return img_