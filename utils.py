import os
import tensorflow as tf

import torch

from torch_to_tf import to_tf_graph
from keras import backend as K
import sys
import tensorflow as tf
from keras.losses import Loss
import yaml
import numpy as np
import cv2
from keras.losses import MeanSquaredError

from multiprocessing import Process, Queue, shared_memory, managers
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


def get_shape(data_config, measure, downsample=1, greyscale=False):
    pref = 'measure_' if measure else 'truth_'
    downsample = downsample if measure else 1
    if greyscale :
        return (data_config[pref + 'height'] // downsample, 
                data_config[pref + 'width'] // downsample, 1)
    else:
        return (data_config[pref + 'height'] // downsample, 
                data_config[pref + 'width'] // downsample,
                data_config[pref + 'channels'])


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
        print('creating lpips loss...')
        lpips_loss = LPIPS(net=model)#.cuda()
        #change to satisfy with torch order : first channels
        sample_input = (torch.randn(batch_size, *shape, requires_grad=False),#.cuda(),
                        torch.randn(batch_size, *shape, requires_grad=False))#.cuda())
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    else:
        print('lpips loss already exists in memory, loading it...')

    stored_lpips = tf.keras.models.load_model(lpips_path + '.pb')
    

    def loss_function(x, y):
        # to NCHW
        x = tf.transpose(x, [0, 3, 1, 2])
        y = tf.transpose(y, [0, 3, 1, 2])
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
    #    for weight, loss in zip(self.loss_weights, self.losses):
    #        print("="*80)
    #        print('loss name', loss.name, ', weight', weight)
    #        print('y_true shape', y_true.shape, 'y_pred shape', y_pred.shape)
    #        print(loss(y_true, y_pred).shape)
       return tf.math.reduce_sum([weight * loss(y_true, y_pred) for weight, loss in zip(self.loss_weights, self.losses)], axis=0)


class LossNamer(Loss):
    def __init__(self, loss, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss = loss

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


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
    return tf.image.ssim(x, y, max_val=1.0)


def psnr(x, y):
    # rescale from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    y = (y + 1) / 2
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




















# class ShmArray(np.ndarray):

#     def __new__(cls, shape, dtype=float, buffer=None, offset=0,
#                 strides=None, order=None, shm=None):
#         obj = super(ShmArray, cls).__new__(cls, shape, dtype,
#                                            buffer, offset, strides,
#                                            order)
#         obj.shm = shm
#         return obj

#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.shm = getattr(obj, 'shm', None)




# class Tee(io.TextIOBase):
#     def __init__(self, *writers):
#         self.writers = writers

#     def write(self, data):
#         for writer in self.writers:
#             writer.write(data)




# def shared_mem_multiprocessing(sequence, workers=8, queue_max_size=32):

#     class ErasingSharedMemory(shared_memory.SharedMemory):

#         def __del__(self):
#             super(ErasingSharedMemory, self).__del__()
#             self.unlink()

#     queue = Queue(maxsize=queue_max_size)
#     manager = managers.SharedMemoryManager()
#     manager.start()

#     def worker(sequence, idxs):
#         for i in idxs:
#             x, y = sequence[i]

#             shm = manager.SharedMemory(size=x.nbytes + y.nbytes)
#             a = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf, offset=0)
#             b = np.ndarray(y.shape, dtype=y.dtype, buffer=shm.buf, offset=x.nbytes)

#             a[:] = x[:]
#             b[:] = y[:]
#             queue.put((a.shape, a.dtype, b.shape, b.dtype, shm.name))
#             shm.close()
#             del shm

#     idxs = np.array_split(np.arange(len(sequence)), workers)
#     args = zip([sequence] * workers, idxs)
#     processes = [Process(target=worker, args=(s, i)) for s, i in args]
#     _ = [p.start() for p in processes]

#     try:
#         for i in range(len(sequence)):
#             x_shape, x_dtype, y_shape, y_dtype, shm_name = queue.get(block=True)
#             existing_shm = ErasingSharedMemory(name=shm_name)
#             x = ShmArray(x_shape, dtype=x_dtype, buffer=existing_shm.buf, offset=0, shm=existing_shm)
#             y = ShmArray(y_shape, dtype=y_dtype, buffer=existing_shm.buf, offset=x.nbytes, shm=existing_shm)
#             yield x, y
#             # Memory will be automatically deleted when gc is triggered
#     finally:
#         print("Closing all the processed")
#         _ = [p.terminate() for p in processes]
#         print("Joining all the processed")
#         _ = [p.join() for p in processes]
#         queue.close()
#         manager.shutdown()
#         manager.join()



