# #############################################################################
# utils.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################


import os
import tensorflow as tf

from keras import backend as K
import sys
import tensorflow as tf
from keras.losses import Loss
import yaml
import numpy as np

from keras.losses import MeanSquaredError

import sys
import io




MAX_UINT16_VAL = 2**16 -1


# from project
def rgb2gray(rgb, weights=np.array([0.299, 0.587, 0.114])):
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
    assert len(weights) == 3
    return np.expand_dims(np.tensordot(rgb, weights, axes=((2,), 0)), -1)


def tf_rgb2gray(rgb, weights=tf.constant([0.299, 0.587, 0.114], dtype=tf.float32)):
    assert len(weights) == 3
    return tf.expand_dims(tf.tensordot(rgb, weights, axes=((2,), 0)), -1)



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


def get_shape(data_config, measure, greyscale=False, resize_input_shape=None):
    pref = 'measure_' if measure else 'truth_'

    num_channels = 1 if greyscale else data_config[pref + 'channels']

    if resize_input_shape and measure:
        return (resize_input_shape[0], resize_input_shape[1], num_channels)  
    else:
        return (data_config[pref + 'height'], data_config[pref + 'width'], num_channels)



def get_config_from_yaml(path):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

# To store the resulted lpips if used in training and testing and not create 2 instances
LPIPS_LOSS = None

def get_lpips_loss(model, shape, batch_size, reduction):
    global LPIPS_LOSS
    if LPIPS_LOSS:
        return LPIPS_LOSS

    if not os.path.isdir('lpips_losses'):
        os.makedirs('lpips_losses')


    def get_lpips_name():
        shape_str = ''
        for s in shape[:-1]:
            shape_str += '_' + str(s)
        channel = shape[-1]
        shape_str = '_' + str(channel) + shape_str
        return 'lpips_' + model + '_shape' + shape_str
    
    lpips_path = os.path.join('lpips_losses', get_lpips_name())

    if not os.path.exists(lpips_path + '.pb'):
        from lpips import LPIPS
        import torch
        from torch_to_tf import to_tf_graph

        print('creating lpips loss...')
        print('target shape :', shape)
        lpips_loss = LPIPS(net=model)#.cuda()
        #change to satisfy with torch order : first channels
        shape = (shape[-1], *shape[:-1])
        print('sample input shape :', shape)
        sample_input = (torch.randn(batch_size, *shape, requires_grad=False),#.cuda(),
                        torch.randn(batch_size, *shape, requires_grad=False))#.cuda())
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    else:
        print('lpips loss already exists in memory, loading it...')

    stored_lpips = tf.keras.models.load_model(lpips_path + '.pb')
    

    def loss_function(x, y):
        # to NCHW
        x = to_channel_first(x)
        y = to_channel_first(y)
        return stored_lpips(input1=x, input2=y)['output']
    
    lpips = loss_function
    lpips = LossNamer(lpips, 'lpips', reduction=reduction)
    
    LPIPS_LOSS = lpips
    
    return lpips



class ChangeLossWeights(tf.keras.callbacks.Callback):
    def __init__(self, weights_factors):
        self.weights_factors = weights_factors

    def on_epoch_end(self, epoch, logs=None):
        for weight, additive_factor in self.weights_factors:
            if weight + additive_factor < 0 :
                print('\nno weight update, current minus value :', weight)
            else:
                K.set_value(weight, weight + additive_factor)


        

# TODO : transform to support dict
class LossCombiner(Loss):
    def __init__(self, losses, loss_weights=None, name='loss_combination', **kwargs):
        super().__init__(name=name, **kwargs)
        if loss_weights:
            assert len(losses) == len(loss_weights), 'the number of weights do not correspond to the number of losses'
            self.loss_weights = loss_weights
        else:
            self.loss_weights = [1] * len(losses)

        self.losses = losses

    def call(self, y_true, y_pred):
        return tf.math.reduce_sum([weight * loss(y_true, y_pred) for weight, loss in zip(self.loss_weights, self.losses)], axis=0)
    
    def get_config(self):
        config = super().get_config()
        config.update({'losses': self.losses, 'loss_weights': self.loss_weights})
        return config


class LossNamer(Loss):
    def __init__(self, loss, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss = loss

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
    
    def get_config(self):
        config = super().get_config()
        config.update({'loss': self.loss})
        return config


# class MSEChannelFirst(Loss):
#     def __init__(self, name='mse'):
#         super().__init__(name=name)

#     def call(self, y_true, y_pred):
#         y_pred = tf.convert_to_tensor(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         return K.mean(tf.math.squared_difference(y_pred, y_true), axis=1)




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

def get_loss_from_name(name_id, distributed_gpu, loss_args=None):
    reduction = tf.keras.losses.Reduction.NONE if distributed_gpu else tf.keras.losses.Reduction.AUTO
    if name_id == 'mse':
        # def custom_mse(y_true, y_pred):
        # return tf.math.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3], keepdims=True)
        if distributed_gpu:
            return LossNamer(lambda x, y: tf.math.reduce_mean(tf.square(x - y), axis=[1, 2, 3], keepdims=True),
                              'mse', reduction=reduction)
        return MeanSquaredError(name='mse', reduction=reduction)
    elif name_id == 'lpips':
        # lpips_model=loss_config['model'], shape=shape, batch_size=batch_size, 
        return get_lpips_loss(reduction=reduction, **loss_args)
    elif name_id == 'ssim':
        return LossNamer(ssim, 'ssim', reduction=reduction)
    elif name_id == 'psnr':
        return LossNamer(psnr, 'psnr', reduction=reduction)
    else:
        raise NotImplementedError('loss not implemented')


def ssim(x, y):
    # rescale from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    y = (y + 1) / 2
    x = tf.clip_by_value(x, 0, 1)
    return tf.image.ssim(x, y, max_val=1.0)


def psnr(x, y):
    # rescale from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    y = (y + 1) / 2
    x = tf.clip_by_value(x, 0, 1)
    return tf.image.psnr(x, y, max_val=1.0)





##########################################################################################
############################ Model optimization visualization ############################
##########################################################################################
def get_size(file_path, unit='bytes'):
    file_size = os.path.getsize(file_path)
    exponents_map = {'bytes': 0, 'kb': 1, 'mb': 2, 'gb': 3}
    if unit not in exponents_map:
        raise ValueError("Must select from \
        ['bytes', 'kb', 'mb', 'gb']")
    else:
        size = file_size / 1024 ** exponents_map[unit]
        return round(size, 3)
    

def get_gzipped_model_size(model, unit='bytes'):
    import os
    import zipfile
    import tempfile
    # It returns the size of the gzipped model in bytes.
    _, keras_file = tempfile.mkstemp('.h5') 

    # model = model.copy()
    for i in range(len(model.weights)):
        model.weights[i]._handle_name = model.weights[i].name + "_" + str(i).zfill(5)

    model.save(keras_file, include_optimizer=False)

    for i in range(len(model.weights)):
        model.weights[i]._handle_name = model.weights[i].name[:-6]


    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)
    return get_size(zipped_file, unit)


class Tee(io.TextIOBase):
    def __init__(self, *writers):
        self.writers = writers

    def write(self, data):
        for writer in self.writers:
            writer.write(data)


# from https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    import numpy as np
    try:
        from keras import backend as K
    except:
        from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes






