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
from keras.losses import MeanSquaredError

from multiprocessing import Process, Queue, shared_memory, managers



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


def get_lpips_loss(config, lpips_model):

    if not os.path.isdir('lpips_losses'):
        os.makedirs('lpips_losses')

    shape = get_shape(config['dataset'], measure=False, greyscale=config['greyscale'])

    def get_lpips_name():
        shape_str = ''
        for s in shape:
            shape_str += '_' + str(s)
        return 'lpips_' + lpips_model + '_shape' + shape_str
    
    lpips_path = os.path.join('lpips_losses', get_lpips_name())

    if not os.path.isfile(lpips_path + '.pb'):
        lpips_loss = LPIPS(net=lpips_model).cuda()
        #change to satisfy with torch order : first channels
        sample_input = (torch.randn(config['batch_size'], *shape, requires_grad=False).cuda(),
                        torch.randn(config['batch_size'], *shape, requires_grad=False).cuda())
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    stored_lpips = tf.keras.models.load_model(lpips_path + '.pb')
    

    # if config['use_crop']:
    #     lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=x, 
    #                                                       input2=y
    #                                                       )['output'])
    # else :
    lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=(x * 2 - 1), input2=(y * 2 - 1))['output'])
    
    return LossNamer(lpips, 'lpips')



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


class MSEChannelFirst(Loss):
    def __init__(self, name='mse'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return K.mean(tf.math.squared_difference(y_pred, y_true), axis=1)




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

def get_loss_from_name(name_id, loss_config, config=None):
    if name_id == 'mse':
        return MeanSquaredError(name='mse')
    elif name_id == 'lpips':
        return get_lpips_loss(config, loss_config['model'])
    else:
        raise NotImplementedError('loss not implemented')






class ShmArray(np.ndarray):

    def __new__(cls, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, shm=None):
        obj = super(ShmArray, cls).__new__(cls, shape, dtype,
                                           buffer, offset, strides,
                                           order)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shm = getattr(obj, 'shm', None)





def shared_mem_multiprocessing(sequence, workers=8, queue_max_size=32):

    class ErasingSharedMemory(shared_memory.SharedMemory):

        def __del__(self):
            super(ErasingSharedMemory, self).__del__()
            self.unlink()

    queue = Queue(maxsize=queue_max_size)
    manager = managers.SharedMemoryManager()
    manager.start()

    def worker(sequence, idxs):
        for i in idxs:
            x, y = sequence[i]

            shm = manager.SharedMemory(size=x.nbytes + y.nbytes)
            a = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf, offset=0)
            b = np.ndarray(y.shape, dtype=y.dtype, buffer=shm.buf, offset=x.nbytes)

            a[:] = x[:]
            b[:] = y[:]
            queue.put((a.shape, a.dtype, b.shape, b.dtype, shm.name))
            shm.close()
            del shm

    idxs = np.array_split(np.arange(len(sequence)), workers)
    args = zip([sequence] * workers, idxs)
    processes = [Process(target=worker, args=(s, i)) for s, i in args]
    _ = [p.start() for p in processes]

    try:
        for i in range(len(sequence)):
            x_shape, x_dtype, y_shape, y_dtype, shm_name = queue.get(block=True)
            existing_shm = ErasingSharedMemory(name=shm_name)
            x = ShmArray(x_shape, dtype=x_dtype, buffer=existing_shm.buf, offset=0, shm=existing_shm)
            y = ShmArray(y_shape, dtype=y_dtype, buffer=existing_shm.buf, offset=x.nbytes, shm=existing_shm)
            yield x, y
            # Memory will be automatically deleted when gc is triggered
    finally:
        print("Closing all the processed")
        _ = [p.terminate() for p in processes]
        print("Joining all the processed")
        _ = [p.join() for p in processes]
        queue.close()
        manager.shutdown()
        manager.join()



