
from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D, Reshape, ZeroPadding2D, Multiply, Cropping2D, CenterCrop
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation

from keras.models import Model
import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_first')
import numpy as np
from keras.losses import MeanSquaredError
from models.unet import u_net

from utils import *
from omegaconf import OmegaConf
import scipy
from scipy.io import loadmat
from PIL import Image
import sys


######################################################################
############################# Separable ##############################
######################################################################



class SeparableLayer(tf.keras.layers.Layer):
    """Layer used for the trainable inversion in FlatNet for the separable case

    Args:
        TODO
    """
    def __init__(self, W1_init, W2_init):
        super(SeparableLayer, self).__init__(name='separable_layer')
        self.W1 = tf.Variable(W1_init, name='camera_inversion_W1')
        self.W2 = tf.Variable(W2_init, name='camera_inversion_W2')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)


    def build(self, input_shape):
        self.in_shape = input_shape
        b, c, h, w = self.in_shape
        print(self.W1.shape)
        print(self.W2.shape)
        assert h == self.W1.shape[1], f"W1 width must be equal to the input height, got {self.W1.shape[1]} and {h}"
        assert w == self.W2.shape[0], f"W1 height must be equal to the input width, got , got {self.W2.shape[0]} and {w}"


    def call(self, x):
        #In NCHW format, o.w. see draft
        # TODO : define best order
        temp = tf.matmul(self.W1, x)
        temp = tf.matmul(temp, self.W2)
        return self.activation(temp)
    




###########################################################################
############################## non-separable ##############################
###########################################################################

@tf.function
def fft_conv2d(x, kernel):
    """ Computes the convolution in the frequency domain given
    Expects input and kernel already in frequency domain

    Args:
        x (tensor): shape (B, C, H, W)
        kernel (tensor): shape (H, W, C)
    """
    kernel = tf.transpose(kernel, (2, 0, 1))

    x = tf.signal.rfft2d(x)
    kernel = tf.signal.rfft2d(kernel)
    
    mult0 = x * kernel

    result = tf.signal.irfft2d(mult0)
    return result


def get_wiener_matrix(psf, gamma: int = 20000):
    """get Wiener matrix of PSF

    Args:
        psf (numpy array): point-spread-function matrix, shape (H, W, C)
        gamma (int, optional): regularization parameter. Defaults to 20000.

    Returns:
        numpy array: wiener filter of psf
    """

    H = np.fft.rfft2(psf, axes=(0, 1))
    H_conj = np.conj(H)

    H_absq = np.abs(H)**2

    res = np.fft.irfft2(H_conj / (gamma + H_absq).astype(np.complex64), axes=(0, 1), s=psf.shape[:2])

    return res.astype(np.float32)




class FTLayer(tf.keras.layers.Layer):
    """Layer used for the trainable inversion in FlatNet for the non-separable case

    Args:
        config (dict) :
        psf_crop (tf.Tensor) :
        mask (TODO): 
    """
    def __init__(self, config, psf, mask=None, name='non_separable_layer', activation='linear',**kwargs):
        super(FTLayer, self).__init__(name=name, **kwargs)
        self.psf = psf
        self.config = dict(config)
        # self.activation = Activation(activation, name=activation)
        self.activation = tf.keras.activations.get(activation)
        

        wiener_crop = get_wiener_matrix(psf, 
                                        gamma=config['wiener_gamma'])
        # TODO : define if better use add_weight

        self.ft_layer = tf.Variable(tf.convert_to_tensor(wiener_crop), name='camera_inversion_ft_layer')

        self.normalizer = tf.Variable([[[[1 / 0.0008]]]], shape=(1, 1, 1, 1), name='camera_inversion_normalizer')

        data_config = config['dataset']
        psf_config = data_config['psf']
        self.pad_x = (psf_config['height'] - psf_config['crop_size_x']) // 2
        self.pad_y = (psf_config['width'] - psf_config['crop_size_y']) // 2


        ft_test = tf.zeros(self.ft_layer.shape)
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
    
        self.mask = tf.Variable(mask, name='camera_inversion_mask') if mask else None


    def build(self, input_shape):
        _, c, h, w = input_shape
        # here pad psf to match input shape
        self.pad_input = None
        self.pad_psf = None
        
        if (h - self.ft_h) < 0 and (w - self.ft_w) < 0:
            # need to pad input
            x_diff = self.ft_h - h
            y_diff = self.ft_w - w
            h_before = x_diff // 2
            h_after = x_diff - h_before
            w_before = y_diff // 2
            w_after = y_diff - w_before
            self.pad_input = ((h_before, h_after), (w_before, w_after))

        elif (h - self.ft_h) >= 0 and (w - self.ft_w) >= 0:
            # need to pad psf
            x_diff = h - self.ft_h
            y_diff = w - self.ft_w
            h_before = x_diff // 2
            h_after = x_diff - h_before
            w_before = y_diff // 2
            w_after = y_diff - w_before
            self.pad_psf = ((h_before, h_after), (w_before, w_after), (0, 0))

            self.in_shape = input_shape
        else:
            raise NotImplementedError('Need to pad PSF and input')
        return
    
    def get_config(self):
        config = super().get_config()

        config.update({
            "psf": self.psf,
            "config": self.config,
            "mask": self.mask
        })
        return config
      

    def call(self, x):
        if self.pad_input:
            x = ZeroPadding2D(padding=self.pad_input)(x)

        curr_ft_layer = self.ft_layer

        if self.pad_psf:
            curr_ft_layer = tf.pad(curr_ft_layer, paddings=self.pad_psf, mode="CONSTANT")

        
        for axis in range(2):
            curr_ft_layer = tf.roll(curr_ft_layer, axis=axis, shift=-(curr_ft_layer.shape[axis] // 2))

        # print('ft_shape', curr_ft_layer.shape)
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

        # x = ZeroPadding2D(padding=self.pad_input)(x)

        x = fft_conv2d(x, curr_ft_layer) * self.normalizer

        # Centre Crop
        # print('x before crop', x.shape)
        # x = x[  :, :, 
        #         self.low_crop_h : self.high_crop_h,
        #         self.low_crop_w : self.high_crop_w]
        # print('x before crop', x.shape)

        x = Cropping2D(cropping=((self.low_crop_h, x.shape[2] - self.high_crop_h),
                                  (self.low_crop_w, x.shape[3] - self.high_crop_w)),
                        data_format='channels_first')(x)

        # print('x after crop', x.shape)
        x = self.activation(x)
        return x
    
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len

class RealFFTConvolve2D:
    def __init__(self, psf, dtype=None, pad=True, norm="ortho", **kwargs):
        """
        Linear operator that performs convolution in Fourier domain, and assumes
        real-valued signals.
        Parameters
        ----------
        psf :py:class:`~numpy.ndarray` or :py:class:`~torch.Tensor`
            2D filter to use.
        dtype : float32 or float64
            Data type to use for optimization.
        """

        # prepare shapes for reconstruction
        # Depth, HWC
        # 0: depth
        # 1: width
        # 2: height
        # 3 : channel
        # [0, 1, 2, 3]

        height, width, channel = psf.shape
        depth = 1

        # cropping / padding indexes
        self._padded_shape = 2 * [width, height] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = list(
            np.r_[depth, self._padded_shape, channel]
        )
        self._start_idx = (self._padded_shape[-3:-1] - [width, height]) // 2
        self._end_idx = self._start_idx + [width, height]
        self.pad = pad  # Whether necessary to pad provided data

        # precompute filter in frequency domain
        self._H = torch.fft.rfft2(
            self._pad(self._psf), norm=norm, dim=(-3, -2), s=self._padded_shape[-3:-1]
        )
        self._Hadj = torch.conj(self._H)
        self._padded_data = torch.zeros(size=self._padded_shape)



    def _crop(self, x):
        return x[:, 
                 self._start_idx[0] : self._end_idx[0], 
                 self._start_idx[1] : self._end_idx[1]]


    def _pad(self, v):
        vpad = torch.zeros(size=self._padded_shape)

        vpad[:, 
             self._start_idx[0] : self._end_idx[0], 
             self._start_idx[1] : self._end_idx[1]] = v
        return vpad


    def convolve(self, x):
        """
        Convolve with pre-computed FFT of provided PSF.
        """
        if self.pad:
            self._padded_data[
                :, self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]
            ] = x
        else:
            self._padded_data[:] = x

        conv_output = torch.fft.ifftshift(
            torch.fft.irfft2(
                torch.fft.rfft2(self._padded_data, dim=(-3, -2)) * self._H, dim=(-3, -2)
            ),
            dim=(-3, -2),
        )

        if self.pad:
            return self._crop(conv_output)
        else:
            return conv_output


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

MAX_UINT8_VAL = 2**8 -1


def get_psf(config):
    data_config = config['dataset']
    psf_config = data_config['psf']
    if data_config['name'] == 'wallerlab':
        psf = (np.array(Image.open(psf_config['path'])) / MAX_UINT16_VAL).astype('float32')
        return rgb2gray(psf) if config['greyscale'] else psf
    
    elif data_config['name'] == 'phlatnet':
        
        psf = extract_bayer(np.load(psf_config['path']))
        # Crop
        crop_top = psf_config['centre_x'] - psf_config['crop_size_x'] // 2
        crop_bottom = psf_config['centre_x'] + psf_config['crop_size_x'] // 2
        crop_left = psf_config['centre_y'] - psf_config['crop_size_y'] // 2
        crop_right = psf_config['centre_y'] + psf_config['crop_size_y'] // 2

        psf_crop = psf[crop_top:crop_bottom, crop_left:crop_right]
        return psf_crop
    
    elif data_config['name'] == 'flatnet':
        raise NotImplementedError('flatnet dataset not implemented yet')
    else:
        raise ValueError(f'Unknown dataset name: {data_config["name"]}, please define how to extract the psf for this dataset')


def get_separable_init_matrices(config):
    if config['dataset']['name'] == 'flatnet':
        d = loadmat(config['dataset']['calibrated_path'])
        phi_l = d['P1gb']
        phi_r = d['Q1gb']
        return phi_l, phi_r
    else:
        raise NotImplementedError('Only flatnet dataset implemented, implement the loading of the left-right matrices for this dataset')



def get_camera_inversion_layer(config, mask=None):
    if config['dataset']['type_mask'] == 'separable':
        phi_l, phi_r = get_separable_init_matrices(config)

        return SeparableLayer(phi_l.T, phi_r)
        
    else :
        psf = get_psf(config) if config['use_psf_init'] else None
        return FTLayer(config, psf=psf, mask=mask)



##############################################################################################################
##############################################################################################################
##############################################################################################################


