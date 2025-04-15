# #############################################################################
# camera_inversion_layers.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

import random
import numpy as np
import cv2
import scipy
from scipy.fftpack import next_fast_len

import tensorflow as tf
from keras.layers import  Cropping2D

import tensorflow_model_optimization as tfmot

from utils.utils import to_channel_first, to_channel_last


############################# Separable ##############################

def get_toeplitz_init(target_shape, slope, is_left=False, seed=1):
    # consider that 
    first_d, second_d = target_shape
    # is_left = first_d >= second_d

    if is_left:
        # cv2 : in (width, height) format
        resized_dims = (int(first_d * slope), first_d)
        np.random.seed(seed)
        arr = np.random.rand(first_d)

        circulant_M = scipy.linalg.circulant(arr)

        resized_M = cv2.resize(circulant_M, resized_dims, interpolation= cv2.INTER_LINEAR)
        random.seed(seed)
        begin_index = random.choice(range(resized_M.shape[1] - second_d))

        cropped_M = resized_M[:, begin_index : begin_index + second_d]

    else : 
        # to get diff matrix for left and right
        seed += 1
        resized_dims = (second_d, int(second_d * slope))
        np.random.seed(seed)
        arr = np.random.rand(second_d)

        circulant_M = scipy.linalg.circulant(arr)

        resized_M = cv2.resize(circulant_M, resized_dims, interpolation= cv2.INTER_LINEAR)
        random.seed(seed)
        begin_index = random.choice(range(resized_M.shape[0] - first_d))

        cropped_M = resized_M[begin_index : begin_index + first_d, :]

    return cropped_M


class SeparableLayer(tf.keras.layers.Layer, tfmot.sparsity.keras.PrunableLayer, tfmot.clustering.keras.ClusterableLayer):
    """Layer used for the trainable inversion in FlatNet for the separable case

    Args:
        W1_init (np array): initial value for W1
        W2_init (np array): initial value for W2
        name (str, optional): name of the layer. Defaults to 'separable_layer'.
    """
    def __init__(self, W1_init, W2_init, name='separable_layer'):
        super(SeparableLayer, self).__init__(name=name)
        self.W1 = tf.Variable(W1_init, name='camera_inversion_W1')
        self.W2 = tf.Variable(W2_init, name='camera_inversion_W2')
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)



    def build(self, input_shape):
        self.in_shape = input_shape
        b, h, w, c = self.in_shape
        # print('W1 shape:', self.W1.shape)
        # print('W2 shape:', self.W2.shape)
        # print('input shape:', self.in_shape)
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



    

############################## non-separable ##############################



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
        psf (numpy array): point-spread-function matrix, shape (H, W, C)
        activation (str, optional): activation function. Defaults to 'linear'.
        gamma (int, optional): regularization parameter. Defaults to 20000.
        pad (bool, optional): whether to pad the input or not to do a valid convolution in frequency domain. Defaults to False.
        name (str, optional): name of the layer. Defaults to 'non_separable_layer'.
    """
    def __init__(self, 
                 psf, 
                 activation='linear', 
                 gamma=20000, 
                 pad=False, 
                 name='non_separable_layer', 
                 **kwargs):
        
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
        
        assert np.all(psf_shape >= in_shape), 'PSF shape must be greater than input shape'

        target_shape = 2 * in_shape - 1 if self.pad else psf_shape

        self._start_idx_input, self._end_idx_input = self._get_pad_idx(img_shape=in_shape, target_shape=target_shape, channel=channel)
        # to pad to efficient computation size
        self._start_idx_psf, self._end_idx_psf = self._get_pad_idx(img_shape=psf_shape, target_shape=target_shape, channel=channel)
        

        
    def _get_pad_idx(self, img_shape, target_shape, channel):
        padded_shape = np.asarray(target_shape)

        padded_shape = np.array([next_fast_len(i) for i in padded_shape])
        # print('padded shape', padded_shape)
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
    
    
    
# #############################################################################
# camera_inversion, numpy implementation
# #############################################################################


import numpy as np


def get_activation(name_id):
    if name_id == 'linear':
        return lambda x: x
    elif name_id == 'relu':
        return lambda x: np.maximum(x, 0)
    elif name_id == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name_id == 'tanh':
        return lambda x: np.tanh(x)
    else:
        raise ValueError('Unknown activation function {}'.format(name_id))
    
    
# TODO: check if this is correct
def to_channel_first(x):
    return np.transpose(x, (0, 3, 1, 2))

def to_channel_last(x):
    return np.transpose(x, (0, 2, 3, 1))



class FTLayerNumpy():
    """Layer used for the trainable inversion in FlatNet for the non-separable case

    Args:
        config (dict) :
        psf_crop (tf.Tensor) :
    """
    def __init__(self, W, normalizer, pad_idx, activation='linear'):
        self.activation = get_activation(activation)
        self.normalizer = normalizer
        self.W = W

        self._start_idx_input, self._end_idx_input = pad_idx['input']
        # to pad to efficient computation size
        self._start_idx_psf, self._end_idx_psf = pad_idx['psf']
        
      

    def _to_ft(self, w):
        w = np.pad(w, ((0,0),
                       (self._start_idx_psf[0], self._end_idx_psf[0]),
                       (self._start_idx_psf[1], self._end_idx_psf[1])), "constant")

        return np.fft.rfft2(w)
    


    def __call__(self, x):        
        x = np.pad(x, ((0,0),
                       (self._start_idx_input[0], self._end_idx_input[0]),
                       (self._start_idx_input[1], self._end_idx_input[1]), (0,0)), "constant")
            
        to_channel_first(x)

        W = self._to_ft(self.W)
        mult = np.fft.rfft2(x) * W
        x = np.fft.ifftshift(np.fft.irfft2(mult),
                                axes=(-2, -1))
        # TODO: check if this is correct
        x = x[:, :, 
              self._start_idx_input[0]:-self._end_idx_input[0], 
              self._start_idx_input[1]:-self._end_idx_input[1]]

        x = x * self.normalizer

        x = to_channel_last(x)
        return self.activation(x)