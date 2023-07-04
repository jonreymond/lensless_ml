# #############################################################################
# camera_inversion_numpy.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
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