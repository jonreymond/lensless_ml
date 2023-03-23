from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation

from keras.models import Model
import tensorflow as tf
import numpy as np

# TODO: remove
import torch
import torch.nn as nn


#####################################################################
###################### Discriminator ################################
#####################################################################
def conv_block(x, filters, kernel_size, strides=(1,1), activation='relu', num_groups=None):
    x = Conv2D(filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same', # TODO : check
                )(x)
    if num_groups:
        x = GroupNormalization(groups=num_groups)(x)
    
    x = Activation(activation=activation)(x)
    return x
    

def get_discriminator(shape, num_groups):
    input = Input(shape=shape, name="input")
    x = input

    filters = [64, 128, 128, 256]
    x = conv_block(x, filters=filters[0], kernel_size=3)
    x = conv_block(x, filters=filters[1], kernel_size=3, stride=2, num_groups=num_groups)
    x = conv_block(x, filters=filters[2], kernel_size=3, num_groups=num_groups)
    x = conv_block(x, filters=filters[3], kernel_size=3, num_groups=num_groups)
    x = GlobalAveragePooling2D()(x)
    x = Conv2D(1, kernel_size=1, padding='same')(x) # TODO: check padding

    return Model(inputs=[input], outputs=[x], name='discriminator')


###########################################
############## fft layer ##################
###########################################

def fft_conv2d(x, kernel):
    """ Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain

    Args:
        x (tensor): shape (B, H, W, Cin)
        kernel (tensor): shape 
    """
    b, h, w, c = kernel.shape
    kernel = tf.reshape(kernel, (b, c, h, w))

    b, h, w, c = x.shape
    x = tf.reshape(x, (b, c, h, w))

    x = tf.signal.rfft2d(x)
    kernel = tf.signal.rfft2d(kernel)

    result = tf.signal.irfft2d(x * kernel)
    b, c, h, w = result.shape
    result = tf.reshape(result, (b, h, w, c))
    return result


def get_wiener_matrix(psf, gamma: int = 20000):
    """get Wiener matrix of PSF

    Args:
        psf (tensor): point-spread-function matrix
        gamma (int, optional): regularization parameter. Defaults to 20000.

    Returns:
        tensor: wiener filter of psf
    """
    H = tf.signal.fft2d(psf)
    H_conj = tf.math.conj(H)
    H_absq = tf.math.abs(H)**2
    res = tf.signal.irfft2d(H_conj / tf.cast(gamma + H_absq, tf.complex128))
    return res


def get_psf(config_psf):
    psf = tf.convert_to_tensor(np.load(config_psf.psf_path), dtype=tf.float32)
    # Crope
    # TODO : check if the //2 is not out (a - b ) //2
    crop_top = config_psf.centre_x - config_psf.crop_size_x // 2
    crop_bottom = config_psf.centre_x + config_psf.crop_size_x // 2
    crop_left = config_psf.centre_y - config_psf.crop_size_y // 2
    crop_right = config_psf.centre_y + config_psf.crop_size_y // 2

    psf_crop = psf[crop_top:crop_bottom, crop_left:crop_right]
    return psf_crop






class FTLayer(tf.keras.layers.Layer):
    def __init__(self, config, psf_crop, mask=None):
        super(FTLayer, self).__init__()

        wiener_crop = get_wiener_matrix(psf_crop, gamma=config['wiener_gamma'])
        # TODO : define if better use add_weight
        self.wiener_crop = tf.Variable(wiener_crop)
        self.normalizer = tf.Variable(1 / 0.0008, shape=(1, 1, 1, 1))

        psf_config = config['psf']
        self.pad_x = (psf_config['height'] - psf_config['crop_size_x']) // 2
        self.pad_y = (psf_config['width'] - psf_config['crop_size_y']) // 2


        ft_test = tf.zeros(self.wiener_crop.shape)
        ft_test = tf.pad(ft_test, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x)), "CONSTANT")
        for axis in range(2):
            ft_test = tf.roll(ft_test, axis=axis, shift=-(ft_test.shape[axis] // 2))

        ft_h, ft_w = ft_test.shape
        img_h = config['image']['height']
        img_w = config['image']['width']
        self.low_crop_h = ft_h // 2 - img_h // 2    
        self.high_crop_h = ft_h // 2 + img_h // 2  

        self.low_crop_w = ft_w // 2 - img_w // 2    
        self.high_crop_w = ft_w // 2 + img_w // 2  

        self.ft_h, self.ft_w = ft_h, ft_w
    
        self.mask = tf.Variable(mask) if mask else None


    def build(self, input_shape):
        self.input_shape = input_shape
        

    def call(self, x):
        # TODO: why not in __init__ ? other reason to not use directly wiener_crop ?
        # pad for fft, way to simplify ?
        ft_layer = 1 * self.wiener_crop
        ft_layer = tf.pad(ft_layer, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x)), "CONSTANT")
        for axis in range(2):
            ft_layer = tf.roll(ft_layer, axis=axis, shift=-(ft_layer.shape[axis] // 2))
        ft_layer = tf.reshape(ft_layer, (1, self.ft_h, self.ft_w, 1))

        x = 0.5 * x + 0.5
        if self.mask:
            x = x * self.mask

        x = fft_conv2d(x, ft_layer) * self.normalizer
        # Centre Crop
        x = x[  :,
                self.low_crop_h : self.high_crop_h,
                self.low_crop_w : self.high_crop_w,
                :]
        return x



# def get_fftlayer(x, config, psf_crop, mask=None):
#     wiener_crop = get_wiener_matrix(psf_crop, gamma=config['wiener_gamma'])
#     wiener_crop = tf.Variable(wiener_crop)
#     normalizer = tf.Variable(1 / 0.0008, shape=(1, 1, 1, 1))

#     if mask:
#         mask = tf.Variable(mask)

#     ##########################
#     ####### forward ##########
#     ##########################
#     ft_layer = wiener_crop * 1

#     psf_config = config['psf']
#     pad_x = psf_config['height'] - psf_config['crop_size_x']
#     pad_y = psf_config['width'] - psf_config['crop_size_y']

#     ft_layer = tf.pad(ft_layer, ((pad_y // 2, pad_y // 2), (pad_x // 2, pad_x // 2)), "CONSTANT")

#     for dim in range(2):
#         ft_layer = tf.roll(ft_layer, axis=dim, shift=-(ft_layer.shape[dim] // 2))

#     # Make 1 x H x W x 1
#     ft_h, ft_w = ft_layer.shape
#     ft_layer = tf.reshape(ft_layer, (1, ft_h, ft_w, 1))

#     img_h = config['image'].height
#     img_w = config['image'].width

#     # Convert to 0...1
#     x = 0.5 * x + 0.5

#     if mask:
#         x = x * mask

#     x = fft_conv2d(x, ft_layer) * normalizer
#     # Centre Crop
#     x = x[  :,
#             ft_h // 2 - img_h // 2 : ft_h // 2 + img_h // 2,
#             ft_w // 2 - img_w // 2 : ft_w // 2 + img_w // 2,
#             :]
#     return x

