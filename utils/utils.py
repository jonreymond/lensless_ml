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
import keras
from keras.losses import MeanSquaredError

import sys
import io

from scipy.io import loadmat
from PIL import Image
import cv2





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
    """
    Convert RGB array to grayscale, Tensorflow version.
    Parameters
    ----------
    rgb : :py:class:`~~tf.Tensor`
        (N_height, N_width, N_channel) image.
    weights : :py:class:`~tf.Tensor`
        [Optional] (3,) weights to convert from RGB to grayscale.
    Returns
    -------
    img :py:class:`~~tf.Tensor`
        Grayscale image of dimension (height, width).
    """
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
    """
    Get the shape of the input data from hydra config file
    """
    pref = 'measure_' if measure else 'truth_'

    num_channels = 1 if greyscale else data_config[pref + 'channels']

    if resize_input_shape and measure:
        return (resize_input_shape[0], resize_input_shape[1], num_channels)  
    else:
        return (data_config[pref + 'height'], data_config[pref + 'width'], num_channels)



def get_config_from_yaml(path):
    """get hydra config from yaml file

    Args:
        path (str): path to yaml file

    Returns:
        _type_: hydra config file
    """
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config



def extract_bayer(arr):
    """Extract bayer pattern from raw image

    Args:
        arr (np.array): raw bayer image of shape (height, width)

    Returns:
        np.array: image of shape (height, width, 4) with bayer pattern
    """
    raw_h, raw_w = arr.shape
    img = np.zeros((raw_h // 2, raw_w // 2, 4), dtype=np.float32)

    img[:, :, 0] = arr[0::2, 0::2]  # r
    img[:, :, 1] = arr[0::2, 1::2]  # gr
    img[:, :, 2] = arr[1::2, 0::2]  # gb
    img[:, :, 3] = arr[1::2, 1::2]  # b
    # tform = skimage.transform.SimilarityTransform(rotation=0.00174)
    # img = skimage.transform.warp(img, tform)
    return img



def to_tf_graph(torch_model, sample_input, store_output):
    """convert a torch model to a tensorflow graph model

    Args:
        torch_model (torch model): torch model to convert
        sample_input (tuple): sample input of the model
        store_output (str): path to store the output
    """
    import torch
    import onnx
    from onnx_tf.backend import prepare


    print('exporting torch to onnx model...')
    torch_model.eval()

    # torch_out = torch_model(sample_input)    

    torch.onnx.export(model = torch_model,               # model being run
                  args = sample_input,                         # model input (or a tuple for multiple inputs)
                  f = store_output,   # where to save the model (can be a file or file-like object)
                  export_params = True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  verbose=True,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input1', 'input2'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   training=torch.onnx.TrainingMode.TRAINING,
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    
    print('onnx model saved to ', store_output)
    onnx_model = onnx.load(store_output)

    # load and change dimensions to be dynamic
    for dim in (0, 2, 3):
        onnx_model.graph.input[0].type.tensor_type.shape.dim[dim].dim_param = '?'
        onnx_model.graph.input[1].type.tensor_type.shape.dim[dim].dim_param = '?'

    # onnx_model, success = onnxsim.simplify(store_output)
    # assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
    # simple_onnx_model_path =  f'{store_output}.simple.onnx'
    # print(f'Generating {simple_onnx_model_path} ...')
    # onnx.save(onnx_model, simple_onnx_model_path)

    # writer.add_onnx_graph(store_output)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(store_output + '.pb') 

    return tf.keras.models.load_model(store_output + '.pb')

#############################################################################################
###################################### Loss functions #######################################
#############################################################################################


# To store the resulted lpips if used in training and testing and not create 2 instances
LPIPS_LOSS = None
def get_lpips_loss(type_model, shape, batch_size, reduction):
    """
    Get the lpips loss function from a torch model and store it in memory to avoid creating it multiple times
    
    Args:
        type_model (str): type of model to use for lpips
        shape (tuple): shape of the input data
        batch_size (int): batch size of the input data
        reduction (str): reduction type of the loss function (tf.keras.losses.Reduction)"""

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
        return 'lpips_' + type_model + '_shape' + shape_str
    
    lpips_path = os.path.join('lpips_losses', get_lpips_name())

    if not os.path.exists(lpips_path + '.pb'):
        from lpips import LPIPS
        import torch


        print('creating lpips loss...')
        print('target shape :', shape)
        lpips_loss = LPIPS(net=type_model)#.cuda()
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



        

# TODO : transform to support dict
class LossCombiner(Loss):
    def __init__(self, losses, loss_weights=None, name='loss_combination', **kwargs):
        """Combines several losses into a single loss function.

        Args:
            losses (list): list of losses to combine.
            loss_weights (list[float32], optional): list of the weights corresponding to each loss function. Defaults to None.
            name (str, optional): name of the loss. Defaults to 'loss_combination'.
        """
        super().__init__(name=name, **kwargs)
        if loss_weights:
            assert len(losses) == len(loss_weights), 'the number of weights do not correspond to the number of loss functions'
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
        """Wraps a loss function to a keras Loss object with a name.
        Args:
            loss (Loss): loss function to wrap.
            name (str): name of the loss."""
        super().__init__(name=name, **kwargs)
        self.loss = loss

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
    
    def get_config(self):
        config = super().get_config()
        config.update({'loss': self.loss})
        return config






def get_loss(name_id, distributed_gpu, loss_args=None):
    reduction = tf.keras.losses.Reduction.NONE if distributed_gpu else tf.keras.losses.Reduction.AUTO
    if name_id == 'mse':
        if distributed_gpu:
            return LossNamer(lambda x, y: tf.math.reduce_mean(tf.square(x - y), axis=[1, 2, 3], keepdims=True),
                              'mse', reduction=reduction)
        else:
            return MeanSquaredError(name='mse', reduction=reduction)
    elif name_id == 'lpips':
        return get_lpips_loss(reduction=reduction, **loss_args)
    else:
        raise NotImplementedError('loss not implemented: ' + name_id)



class DistributedLossCombiner(Loss):
    def __init__(self, losses, loss_weights=None, name='total', global_batch_size=None, **kwargs):
        """Combines several losses into a single loss function, but reduces manually the loss in case of distributed gpu training.

        Args:
            losses (list): list of losses to combine.
            loss_weights (list[float32], optional): list of the weights corresponding to each loss function. Defaults to None.
            name (str, optional): name of the loss. Defaults to 'loss_combination'.
        """
        self.losses = losses
        self.loss_weights = loss_weights
        self.global_batch_size = global_batch_size
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)

    def call(self, y_true, y_pred):
        loss = tf.add_n([weight * tf.nn.compute_average_loss(loss(y_true, y_pred), global_batch_size=self.global_batch_size) for weight, loss in zip(self.loss_weights, self.losses)])

        # if model_losses:
        #     loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'losses': self.losses,
            'loss_weights': self.loss_weights,
            'global_batch_size': self.global_batch_size
        })
        return config


class DistributedLoss(Loss):
    def __init__(self, loss, name, global_batch_size, **kwargs):
        """Wraps a loss function to a keras Loss object with a name, but reduces manually the loss in case of distributed gpu training.

        Args:
            loss (Loss): loss function to wrap.
            name (str): name of the loss."""
        
        self.loss = loss
        self.global_batch_size = global_batch_size
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        
    
    def call(self, y_true, y_pred):
        per_sample_loss = self.loss(y_true, y_pred)
        loss = tf.nn.compute_average_loss(per_sample_loss, global_batch_size=self.global_batch_size)
        # if model_losses:
        #     loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'loss': self.loss,
            'global_batch_size': self.global_batch_size
        })
        return config
    
#############################################################################################
###################################### Metric functions #####################################
#############################################################################################





def ssim(x, y):
    """computes the structural similarity index between two images.

    Args:
        x (tf.Tensor): first image of shape [batch_size, height, width, channels]
        y (tf.Tensor): second image of shape [batch_size, height, width, channels]

    Returns:
        tf.Tensor: structural similarity index between x and y of shape [batch_size]
    """
    # rescale from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    y = (y + 1) / 2
    x = tf.clip_by_value(x, 0, 1)
    return tf.image.ssim(x, y, max_val=1.0)


def psnr(x, y):
    """computes the peak signal-to-noise ratio between two images.

    Args:
        x (tf.Tensor): first image of shape [batch_size, height, width, channels]
        y (tf.Tensor): second image of shape [batch_size, height, width, channels]

    Returns:
        tf.Tensor: peak signal-to-noise ratio between x and y of shape [batch_size]
    """
    # rescale from [-1, 1] to [0, 1]
    x = (x + 1) / 2
    y = (y + 1) / 2
    x = tf.clip_by_value(x, 0, 1)
    return tf.image.psnr(x, y, max_val=1.0)


def get_metric(name_id, distributed_gpu=False, loss_args=None):
    """get metric from name

    Args:
        name_id (str): name of desired metric
        distributed_gpu (bool, optional): whether to use distributed gpu metric. Defaults to False.
        loss_args (dict, optional): optional parameters for the corresponding loss (for LPIPS in particular). Defaults to None.

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """
    if distributed_gpu:
        raise NotImplementedError('distributed gpu metric not implemented')
    
    
    if name_id == 'ssim':
        return keras.metrics.MeanMetricWrapper(ssim, name='ssim')
    elif name_id == 'psnr':
        return keras.metrics.MeanMetricWrapper(psnr, name='psnr') 
    elif name_id == 'lpips':
        reduction = tf.keras.losses.Reduction.NONE if distributed_gpu else tf.keras.losses.Reduction.AUTO
        return get_lpips_loss(reduction=reduction, **loss_args)
    else:
        return keras.metrics.MeanMetricWrapper(keras.metrics.get(name_id), name=name_id)


##########################################################################################
############################ Model optimization visualization ############################
##########################################################################################


    




class Tee(io.TextIOBase):
    def __init__(self, *writers):
        self.writers = writers

    def write(self, data):
        for writer in self.writers:
            writer.write(data)


# from https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(batch_size, model):
    import numpy as np

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






def get_separable_init_matrices(config):
    if config['name'] == 'flatnet':
        d = loadmat(config['calibrated_path'])
        phi_l = d['P1gb']
        phi_r = d['Q1gb']
        return phi_l, phi_r
    elif config['name'] == 'wallerlab':
        phi_l = np.load(config['random_toeplitz_path']['phi_l']).astype('float32')
        phi_r = np.load(config['random_toeplitz_path']['phi_r']).astype('float32')
        return phi_l, phi_r
    else:
        raise NotImplementedError('Only flatnet dataset implemented, implement the loading of the left-right matrices for this dataset')



MAX_UINT8_VAL = 2**8 -1

def get_psf(data_config, input_shape=None):
    psf_config = data_config['psf']
    if data_config['name'] == 'wallerlab':
        psf = (np.array(Image.open(psf_config['path'])) / MAX_UINT8_VAL).astype('float32')
        return psf
    
    elif data_config['name'] == 'phlatnet':
        psf = np.load(psf_config['path']).astype('float32')

        # psf = psf[:: data_config['downsample'], :: data_config['downsample'], :]
        if input_shape is not None:
            psf = cv2.resize(psf, (input_shape[1], input_shape[0]))
            print('psf shape', psf.shape, 'psf type', psf.dtype)
        return psf
    
    elif data_config['name'] == 'flatnet':
        raise NotImplementedError('flatnet dataset not implemented yet')
    else:
        raise ValueError(f'Unknown dataset name: {data_config["name"]}, please define how to extract the psf for this dataset')