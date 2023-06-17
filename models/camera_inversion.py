
from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D, Reshape, ZeroPadding2D, Multiply, Cropping2D, CenterCrop
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation

from keras.models import Model
import tensorflow as tf

import numpy as np
from keras.losses import MeanSquaredError
from models.unet import u_net

from utils import *
from omegaconf import OmegaConf
import scipy
from scipy import fft
from scipy.fftpack import next_fast_len
from scipy.io import loadmat
from PIL import Image
import sys

import tensorflow_model_optimization as tfmot

######################################################################
############################# Separable ##############################
######################################################################



class SeparableLayer(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer, tfmot.clustering.keras.ClusterableLayer):
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
        b, h, w, c = self.in_shape
        print('W1 shape:', self.W1.shape)
        print('W2 shape:', self.W2.shape)
        print('input shape:', self.in_shape)
        assert h == self.W1.shape[0], f"W1 width must be equal to the input height, got {self.W1.shape[0]} and {h}"
        assert w == self.W2.shape[0], f"W1 height must be equal to the input width, got , got {self.W2.shape[0]} and {w}"


    def get_config(self):
        config = super().get_config()

        config.update({
            "W1": self.W1,
            "W2": self.W2
        })
        return config
    

    def call(self, x):
        #In NCHW format: tf.matmul inner-most 2 dimensions
        x = to_channel_first(x)
        x = tf.matmul(self.W1, x, transpose_a=True)
        x = tf.matmul(x, self.W2)
        x = to_channel_last(x)

        return self.activation(x)
    

    def get_list_weights(self):
        return [self.W1, self.W2]

    def set_list_weights(self, list_weights):
        self.W1 = list_weights[0]
        self.W2 = list_weights[1]

    def get_prunable_weights(self):
        return [self.W1, self.W2]
    
    def get_clusterable_weights(self):
        return [('W1', self.W1), ('W2', self.W2)]



    
    




###########################################################################
############################## non-separable ##############################
###########################################################################

# @tf.function
# def fft_conv2d(x, kernel):
#     """ Computes the convolution in the frequency domain given
#     Expects input and kernel already in frequency domain

#     Args:
#         x (tensor): shape (B, C, H, W)
#         kernel (tensor): shape (H, W, C)
#     """
#     kernel = tf.transpose(kernel, (2, 0, 1))

#     x = tf.signal.rfft2d(x)
#     kernel = tf.signal.rfft2d(kernel)
    
#     mult0 = x * kernel

#     result = tf.signal.irfft2d(mult0)
#     return result


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



    



class FTLayer(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer, tfmot.clustering.keras.ClusterableLayer):
    """Layer used for the trainable inversion in FlatNet for the non-separable case

    Args:
        config (dict) :
        psf_crop (tf.Tensor) :
    """
    def __init__(self, psf, activation='linear', gamma=20000, pad=False,name='non_separable_layer', **kwargs):
        super(FTLayer, self).__init__(name=name, **kwargs)
        self.psf = psf
        self.pad = pad
        self.activation = tf.keras.activations.get(activation)
        self.gamma = gamma

        self.psf_shape = psf.shape

        wiener_crop = tf.convert_to_tensor(get_wiener_matrix(psf, gamma=self.gamma))
        wiener_crop = tf.transpose(wiener_crop, (2, 0, 1))
        
        self.W = tf.Variable(wiener_crop, name='camera_inversion_W')

        self.normalizer = tf.Variable([[[[1 / 0.0008]]]], shape=(1, 1, 1, 1), name='camera_inversion_normalizer')



    def build(self, input_shape):
        channel = input_shape[3]
        
        psf_shape = np.asarray(self.psf_shape[:2])
        in_shape = np.asarray(input_shape[1:3])
        print('psf shape', psf_shape)
        print('in shape', in_shape)
        
        assert np.all(psf_shape >= in_shape), 'PSF shape must be greater than input shape'

        target_shape = 2 * in_shape - 1 if self.pad else psf_shape

        self._start_idx_input, self._end_idx_input = self._get_pad_idx(img_shape=in_shape, target_shape=target_shape, channel=channel)
        # to pad to efficient computation size
        self._start_idx_psf, self._end_idx_psf = self._get_pad_idx(img_shape=psf_shape, target_shape=target_shape, channel=channel)
        

        
    def _get_pad_idx(self, img_shape, target_shape, channel):
        padded_shape = np.asarray(target_shape)

        padded_shape = np.array([next_fast_len(i) for i in padded_shape])
        print('padded shape', padded_shape)
        padded_shape = list(np.r_[padded_shape, channel])

        start_idx = (padded_shape[0 : 2] - img_shape) // 2

        end_idx = start_idx + (padded_shape[0 : 2] - img_shape) % 2
        return start_idx, end_idx
        
    
    def get_config(self):
        config = super().get_config()

        config.update({
            "psf": self.psf,
            "activation": self.activation,
            "pad": self.pad,
            "gamma": self.gamma
        })
        return config
      

    def _to_ft(self, w):
        w = tf.pad(w, ((0,0),
                       (self._start_idx_psf[0], self._end_idx_psf[0]),
                       (self._start_idx_psf[1], self._end_idx_psf[1])), "CONSTANT")
        

        return tf.signal.rfft2d(w)


    def call(self, x):        
        x = tf.pad(x, ((0,0),
                       (self._start_idx_input[0], self._end_idx_input[0]),
                       (self._start_idx_input[1], self._end_idx_input[1]), (0,0)), "CONSTANT")
        
        
        x = to_channel_first(x)

        
        W = self._to_ft(self.W)

        mult = tf.signal.rfft2d(x) * W

        x = tf.signal.ifftshift(tf.signal.irfft2d(mult),
                                axes=(-2, -1))
        
        x = Cropping2D(cropping=((self._start_idx_input[0], self._end_idx_input[0]),
                                     (self._start_idx_input[1], self._end_idx_input[1])),
                                     data_format='channels_first')(x)

        x = x * self.normalizer

        x = to_channel_last(x)

        return self.activation(x)
    
    def get_list_weights(self):
        return [self.W, self.normalizer]
    
    def set_list_weights(self, list_weights):
        self.W = list_weights[0]
        self.normalizer = list_weights[1]

    def get_prunable_weights(self):
        return [self.W]
    
    def get_clusterable_weights(self):
        return [('W', self.W)]
    


##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

MAX_UINT8_VAL = 2**8 -1
MAX_UINT16_VAL = 2**16 -1

def get_psf(data_config, input_shape=None):
    psf_config = data_config['psf']
    if data_config['name'] == 'wallerlab':
        psf = (np.array(Image.open(psf_config['path'])) / MAX_UINT8_VAL).astype('float32')
        return psf
    
    elif data_config['name'] == 'phlatnet':
        psf = np.load(psf_config['path'])

        # psf = psf[:: data_config['downsample'], :: data_config['downsample'], :]
        if input_shape is not None:
            psf = cv2.resize(psf, (input_shape[1], input_shape[0]))
            print('psf shape', psf.shape, 'psf type', psf.dtype)

        return psf
    
    elif data_config['name'] == 'flatnet':
        raise NotImplementedError('flatnet dataset not implemented yet')
    else:
        raise ValueError(f'Unknown dataset name: {data_config["name"]}, please define how to extract the psf for this dataset')


def get_separable_init_matrices(config):
    if config['name'] == 'flatnet':
        d = loadmat(config['calibrated_path'])
        phi_l = d['P1gb']
        phi_r = d['Q1gb']
        return phi_l, phi_r
    else:
        raise NotImplementedError('Only flatnet dataset implemented, implement the loading of the left-right matrices for this dataset')



# def get_camera_inversion_layer(data_config, input_shape=None, camera_inversion_args=None):
#     if camera_inversion_args['use_random_init']:
#         raise NotImplementedError('Random init not implemented yet')
    
#     camera_inversion_args = dict(camera_inversion_args)
#     camera_inversion_args.pop('use_psf_init')

#     if data_config['type_mask'] == 'separable':
#         phi_l, phi_r = get_separable_init_matrices(data_config)

#         return SeparableLayer(phi_l.T, phi_r)
        
#     else :
#         psf = get_psf(data_config, input_shape=input_shape)
#         # TODO : add rgb2gray
#         return FTLayer(**camera_inversion_args, psf=psf)



##############################################################################################################
##############################################################################################################
##############################################################################################################


