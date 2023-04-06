
from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D, Reshape, ZeroPadding2D, Multiply
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation

from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.losses import MeanSquaredError
from models.unet import u_net

from utils import *






######################################################################
############################# Separable ##############################
######################################################################



class SeparableLayer(tf.keras.layers.Layer):
    """Layer used for the trainable inversion in FlatNet for the separable case

    Args:
        TODO
    """
    def __init__(self, W1_init, W2_init):
        self.W1 = tf.Variable(W1_init)
        self.W2 = tf.Variable(W2_init)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)


    def build(self, input_shape):
        self.input_shape = input_shape
        b, h, w, c = input_shape
        assert h == self.W1.shape[2], "W1 width must be equal to the input height, got " + str(self.W1.shape[2]) + " and " +str(h)
        assert w == self.W1.shape[1], "W1 height must be equal to the input width, got " + str(self.W1.shape[1]) + " and " +str(w)


    def call(self, x):
        #In NCHW format, o.w. see draft
        # TODO : define best order
        temp = tf.matmul(self.W1, x)
        temp = tf.matmul(temp, self.W2)
        return self.activation(temp)
    




###########################################################################
############################## non-separable ##############################
###########################################################################


def fft_conv2d(x, kernel):
    """ Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain

    Args:
        x (tensor): shape (C, B, H, W)
        kernel (tensor): shape (H, W, C)
    """
    
    x = to_channel_last(x)
    print('x shape before :', x.shape)
    x = tf.signal.rfft2d(x)
    print('x shape after :', x.shape)
    
    print('kernel shape before :', kernel.shape)
    kernel = tf.signal.rfft2d(kernel)
    
    kernel = tf.expand_dims(kernel, axis=0)
    print('kernel shape after:', kernel.shape)
    result = tf.signal.irfft2d(x * kernel)

    return to_channel_first(result)


def get_wiener_matrix(psf, gamma: int = 20000):
    """get Wiener matrix of PSF

    Args:
        psf (tensor): point-spread-function matrix
        gamma (int, optional): regularization parameter. Defaults to 20000.

    Returns:
        tensor: wiener filter of psf
    """
    H = tf.signal.rfft2d(psf)
    H_conj = tf.math.conj(H)
    H_absq = tf.math.abs(H)**2

    res = tf.signal.irfft2d(H_conj / tf.cast(gamma + H_absq, tf.complex64))
    return res


def get_psf_cropped(psf_config):
    """Load and crop the psf

    Args:
        config_psf (dict): config of psf

    Returns:
        tensor: cropped psf tensor
    """
    psf = extract_bayer(np.load(psf_config['path']))
    psf = tf.convert_to_tensor(psf, dtype=tf.float32)
    # Crop
    crop_top = psf_config['centre_x'] - psf_config['crop_size_x'] // 2
    crop_bottom = psf_config['centre_x'] + psf_config['crop_size_x'] // 2
    crop_left = psf_config['centre_y'] - psf_config['crop_size_y'] // 2
    crop_right = psf_config['centre_y'] + psf_config['crop_size_y'] // 2

    psf_crop = psf[crop_top:crop_bottom, crop_left:crop_right]
    return psf_crop


class FTLayer(tf.keras.layers.Layer):
    """Layer used for the trainable inversion in FlatNet for the non-separable case

    Args:
        config (dict) :
        psf_crop (tf.Tensor) :
        mask (TODO): 
    """
    def __init__(self, config, psf_cropped, mask=None):
        super(FTLayer, self).__init__()

        wiener_crop = get_wiener_matrix(psf_cropped, gamma=config['wiener_gamma'])
        # TODO : define if better use add_weight

        self.wiener_crop = tf.Variable(wiener_crop)


        self.normalizer = tf.Variable([[[[1 / 0.0008]]]], shape=(1, 1, 1, 1))

        data_config = config['dataset']
        psf_config = data_config['psf']
        self.pad_x = (psf_config['height'] - psf_config['crop_size_x']) // 2
        self.pad_y = (psf_config['width'] - psf_config['crop_size_y']) // 2


        ft_test = tf.zeros(self.wiener_crop.shape)
        ft_test = tf.pad(ft_test, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)), "CONSTANT")
        for axis in range(2):
            ft_test = tf.roll(ft_test, axis=axis, shift=-(ft_test.shape[axis] // 2))

        ft_h, ft_w, _ = ft_test.shape

        img_h = data_config['truth_height']
        img_w = data_config['truth_width']
        self.low_crop_h = ft_h // 2 - img_h // 2    
        self.high_crop_h = ft_h // 2 + img_h // 2  

        self.low_crop_w = ft_w // 2 - img_w // 2    
        self.high_crop_w = ft_w // 2 + img_w // 2  

        self.ft_h, self.ft_w = ft_h, ft_w
    
        self.mask = tf.Variable(mask) if mask else None


    def build(self, input_shape):
        # self.input_shape = input_shape
        _, c, h, w = input_shape

        x_diff = self.ft_h - h
        y_diff = self.ft_w - w
        h_before = x_diff // 2
        h_after = x_diff - h_before
        w_before = y_diff // 2
        w_after = y_diff - w_before
        self.pad_input = ((h_before, h_after), (w_before, w_after))
        return
        

    def call(self, x):
        # TODO: why not in __init__ ? other reason to not use directly wiener_crop ?
        # pad for fft, way to simplify ?
        ft_layer = 1 * self.wiener_crop
        # print('ft layer before:', ft_layer.shape)
        # print((self.pad_y, self.pad_y), (self.pad_x, self.pad_x))
        ft_layer = tf.pad(ft_layer, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)), "CONSTANT")
        for axis in range(2):
            ft_layer = tf.roll(ft_layer, axis=axis, shift=-(ft_layer.shape[axis] // 2))

        # print('ft layer after:', ft_layer.shape)
        # print(ft_layer.shape)
        # TODO : change # channels 
        # ft_layer = tf.reshape(ft_layer, (1, 1, self.ft_h, self.ft_w))
        # ft_layer =  Reshape((self.ft_h, self.ft_w, 1))(ft_layer)
        # sys.exit()

        # to range [0, 1]
        x = 0.5 * x + 0.5

        if self.mask:
            x = x * self.mask

        x = ZeroPadding2D(padding=self.pad_input)(x)

        x = fft_conv2d(x, ft_layer) * self.normalizer

        # Centre Crop
        x = x[  :, :, 
                self.low_crop_h : self.high_crop_h,
                self.low_crop_w : self.high_crop_w]

        return x
    



def get_inversion_model(config, input_shape):
    

    input = Input(shape=input_shape)

    psf_cropped = get_psf_cropped(config['dataset']['psf'])
    mask = np.load(config['dataset']['mask_path'], np.float32) if config['use_mask'] else None
    x = FTLayer(config, psf_cropped=psf_cropped, mask=mask)(input)
    enhancer_model = u_net(x.shape[1:], **config['model']['enhancer']['unet32'])
    output = enhancer_model(x)
    return Model(inputs=[input], outputs=[output], name='Generator')






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