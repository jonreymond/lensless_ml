import os
import tensorflow as tf
from lpips import LPIPS
import torch
from torch_to_tf import to_tf_graph
from keras import backend as K
import sys
import tensorflow as tf
from keras.losses import Loss
import yaml
import numpy as np
import cv2



MAX_UINT16_VAL = 2**16 -1


# from project
def rgb2gray(rgb, weights=None):
    """
    Convert RGB array to grayscale.
    Parameters
    ----------
    rgb : :py:class:`~numpy.ndarray`
        (N_height, N_width, N_channel) image.
    weights : :py:class:`~numpy.ndarray`
        [Optional] (3,) weights to convert from RGB to grayscale.
    Returns
    -------
    img :py:class:`~numpy.ndarray`
        Grayscale image of dimension (height, width).
    """
    if weights is None:
        weights = np.array([0.299, 0.587, 0.114])
    assert len(weights) == 3
    return np.expand_dims(np.tensordot(rgb, weights, axes=((2,), 0)), -1)



# TODO : how to use @tf.function with tf and keras same time
def to_channel_last(x):
    """from NCHW to NHWC format

    Args:
        x (tf tensor): input in NCHW format

    Returns:
        tf tensor: output in NHWC format
    """
    return tf.keras.layers.Permute([2, 3, 1])(x)
    
    
def to_channel_first(x):
    """from NHWC to NCHW format

    Args:
        x (tf tensor): input in NHWC format

    Returns:
        tf tensor: output in NCHW format
    """
    return tf.keras.layers.Permute([3, 1, 2])(x)


def get_shape(data_config, measure, greyscale=False):
    pref = 'measure_' if measure else 'truth_'
    if greyscale :
        return (1, 
                data_config[pref + 'height'], 
                data_config[pref + 'width'])
    else:
        return (data_config[pref + 'channels'], 
                data_config[pref + 'height'], 
                data_config[pref + 'width'])


def get_config_from_yaml(path):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_lpips_loss(config):

    if not os.path.isdir('lpips_losses'):
        os.makedirs('lpips_losses')

    shape = get_shape(config['dataset'], measure=False, greyscale=config['greyscale'])

    def get_lpips_name():
        shape_str = ''
        for s in shape:
            shape_str += '_' + str(s)
        return 'lpips_' + config['lpips_model'] + '_shape' + shape_str
    
    lpips_path = os.path.join('lpips_losses', get_lpips_name())

    if not os.path.isfile(lpips_path + '.pb'):
        lpips_loss = LPIPS(net=config['lpips_model']).cuda()
        #change to satisfy with torch order : first channels
        sample_input = (torch.randn(config['batch_size'], *shape, requires_grad=False).cuda(),
                        torch.randn(config['batch_size'], *shape, requires_grad=False).cuda())
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    stored_lpips = tf.keras.models.load_model(lpips_path + '.pb')
    

    if config['use_crop']:
        lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=x, 
                                                          input2=y
                                                          )['output'])
    else :
        lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=(x * 2 - 1), input2=(y * 2 - 1))['output'])
    
    return LossNamer(lpips, 'lpips')



class ChangeLossWeights(tf.keras.callbacks.Callback):
    def __init__(self, alpha_minus, alpha_plus, factor):
        self.alpha_minus =alpha_minus
        self.alpha_plus = alpha_plus
        self.fact = factor

    def on_epoch_end(self, epoch, logs=None):
        if (self.alpha_minus - self.fact) < 0:
            print('\nno weight update, current minus value :', self.alpha_minus.numpy())
        else:
            K.set_value(self.alpha_plus, self.alpha_plus + self.fact)
            K.set_value(self.alpha_minus, self.alpha_minus - self.fact)
        

# TODO : transform to support dict
class LossCombiner(Loss):
    def __init__(self, losses, loss_weights=None, name='loss_combination'):
        super().__init__(name=name)
        if loss_weights:
            assert len(losses) == len(loss_weights), 'the number of weights do not correspond to the number of losses'
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1] * len(losses)

        self.losses = losses

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum([weight * loss(y_true, y_pred) for weight, loss in zip(self.loss_weights, self.losses)])


class LossNamer(Loss):
    def __init__(self, loss, name):
        super().__init__(name=name)
        self.loss = loss

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


def extract_bayer(arr):
    raw_h, raw_w = arr.shape
    img = np.zeros((raw_h // 2, raw_w // 2, 4), dtype=np.float32)

    img[:, :, 0] = arr[0::2, 0::2]  # r
    img[:, :, 1] = arr[0::2, 1::2]  # gr
    img[:, :, 2] = arr[1::2, 0::2]  # gb
    img[:, :, 3] = arr[1::2, 1::2]  # b
    # tform = skimage.transform.SimilarityTransform(rotation=0.00174)
    # im1=skimage.transform.warp(im1,tform)
    return img



        