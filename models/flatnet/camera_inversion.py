
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



######################################################################
############################# Separable ##############################
######################################################################



class SeparableLayer(tf.keras.layers.Layer):
    """Layer used for the trainable inversion in FlatNet for the separable case

    Args:
        TODO
    """
    def __init__(self, W1_init, W2_init):
        super(SeparableLayer, self).__init__()
        self.W1 = tf.Variable(W1_init)
        self.W2 = tf.Variable(W2_init)
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
        tensor: cropped psf numpy
    """
    psf = extract_bayer(np.load(psf_config['path']))
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
        self.psf_cropped = psf_cropped
        self.config = config

 
        wiener_crop = get_wiener_matrix(tf.convert_to_tensor(psf_cropped, dtype=tf.float32), 
                                        gamma=config['wiener_gamma'])
        # TODO : define if better use add_weight

        self.ft_layer = tf.Variable(wiener_crop)

        # print('ft layer shape', self.ft_layer.shape)


        self.normalizer = tf.Variable([[[[1 / 0.0008]]]], shape=(1, 1, 1, 1))

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
    
        self.mask = tf.Variable(mask) if mask else None


    def build(self, input_shape):
        # self.input_shape = input_shape
        _, c, h, w = input_shape
        x_diff = h - self.ft_h
        y_diff = w - self.ft_w
        h_before = x_diff // 2
        h_after = x_diff - h_before
        w_before = y_diff // 2
        w_after = y_diff - w_before
        self.pad_input = ((h_before, h_after), (w_before, w_after), (0,0))
        self.in_shape = input_shape

        # to pad x 
        # x_diff = self.ft_h - h
        # y_diff = self.ft_w - w
        # h_before = x_diff // 2
        # h_after = x_diff - h_before
        # w_before = y_diff // 2
        # w_after = y_diff - w_before
        # self.pad_input = ((h_before, h_after), (w_before, w_after))
        # self.in_shape = input_shape
        return
    
    def get_config(self):
        config = super().get_config()

        config.update({
            "psf_cropped": self.psf_cropped,
            "config": OmegaConf.to_object(self.config),
            "mask": self.mask
        })
        return config
      

    def call(self, x):
        # TODO: why not in __init__ ? other reason to not use directly wiener_crop ?
        # pad for fft, way to simplify ?

        # print('ft layer before:', ft_layer.shape)
        # print((self.pad_y, self.pad_y), (self.pad_x, self.pad_x))
        # curr_ft_layer = tf.pad(self.ft_layer, ((self.pad_y, self.pad_y), (self.pad_x, self.pad_x), (0, 0)), "CONSTANT")
        # print('x shape input', x.shape)

        curr_ft_layer = tf.pad(self.ft_layer, paddings=self.pad_input, mode="CONSTANT")
        # print('ft_shape', curr_ft_layer.shape)
        
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
        return x
    



def get_inversion_model(config, input_shape):
    
    input = Input(shape=input_shape)

    
    if config['dataset']['type_mask'] == 'separable':
        d = scipy.io.loadmat(config['dataset']['calibrated_path'])
        phi_l = np.zeros((500, 256))
        phi_r = np.zeros((620, 256))
        phi_l[:,:] = d['P1gb']
        phi_r[:,:] = d['Q1gb']
        phi_l = phi_l.astype('float32')
        phi_r = phi_r.astype('float32') 
        x = SeparableLayer(phi_l.T, phi_r)(input)
        
    else :
        psf_cropped = get_psf_cropped(config['dataset']['psf'])
        mask = np.load(config['dataset']['mask_path'], np.float32) if config['use_mask'] else None
        x = FTLayer(config, psf_cropped=psf_cropped, mask=mask)(input)
        # output = tf.repeat(tf.expand_dims(tf.math.reduce_sum(x, axis=1), 1), repeats=3, axis=1)


    print('before unet', x.shape[1:])
    enhancer_model = u_net(x.shape[1:], **config['model']['enhancer']['unet32'])
    output = enhancer_model(x)
    return Model(inputs=[input], outputs=[output], name='Generator')

