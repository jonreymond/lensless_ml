import os
import tensorflow as tf
from lpips import LPIPS
import torch
from torch_to_tf import to_tf_graph
from keras import backend as K
import sys
import tensorflow as tf
from keras.losses import Loss



def to_channel_last(x):
    """from NCHW to NHWC format

    Args:
        x (tf tensor): input in NCHW format

    Returns:
        tf tensor: output in NHWC format
    """
    # n, c, h, w = x.shape
    # return tf.keras.layers.Reshape((h, w, c))(x)
    # return tf.reshape(x, (n, h, w, c))
    return tf.keras.layers.Permute([2, 3, 1])(x)
    
    

def to_channel_first(x):
    """from NHWC to NCHW format

    Args:
        x (tf tensor): input in NHWC format

    Returns:
        tf tensor: output in NCHW format
    """
    # n, h, w, c = x.shape
    # return tf.keras.layers.Reshape((c, h, w))(x)
    return tf.keras.layers.Permute([3, 1, 2])(x)
    # return tf.transpose(images_nhwc, [0, 3, 1, 2])



def get_lpips_loss(config, data_spec):
    dataset_config = config['dataset']
    train_config = config['train_params']
    if not os.path.isdir('lpips_losses'):
        os.makedirs('lpips_losses')

    def get_lpips_name(config, data_spec):
        shape_str = ''
        for s in data_spec['shape']:
            shape_str += '_' + str(s)
        return 'lpips_' + config['train_params']['lpips_model'] + '_shape' + shape_str
    
    lpips_path = os.path.join('lpips_losses', get_lpips_name(config, data_spec))

    if not os.path.isfile(lpips_path + '.pb'):
        lpips_loss = LPIPS(net=train_config ['lpips_model']).cuda()
        #change to satisfy with torch order : first channels
        sample_input = (torch.randn(dataset_config['batch_size'], *data_spec['shape'], requires_grad=False).cuda(),
                        torch.randn(dataset_config['batch_size'], *data_spec['shape'], requires_grad=False).cuda())
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    stored_lpips = tf.keras.models.load_model(lpips_path + '.pb')
    
    # lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=tf.transpose(x, perm=[0, 3, 1, 2]), input2=tf.transpose(y, perm=[0, 3, 1, 2]))['output'])

    if 'crop' in data_spec:
        lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=x, 
                                                          input2=y
                                                          )['output'])
    else :
        lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=(x * 2 - 1), input2=(y * 2 - 1))['output'])
    
    return LossNamer(lpips, 'lpips')



class ChangeLossWeights(tf.keras.callbacks.Callback):
    def __init__(self, alpha_minus, alpha_plus, factor):
        self.alpha_minus =alpha_minus
        self.alpha_plus = alpha_plus
        self.fact = factor

    def on_epoch_end(self, epoch, logs=None):
        if (self.alpha_minus - self.fact) < 0:
            print('\nno weight update, current minus value :', self.alpha_minus.numpy())
        else:
            K.set_value(self.alpha_plus, self.alpha_plus + self.fact)
            K.set_value(self.alpha_minus, self.alpha_minus - self.fact)
        

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
        return sum([weight * loss(y_true, y_pred) for weight, loss in zip(self.loss_weights, self.losses)])


class LossNamer(Loss):
    def __init__(self, loss, name):
        super().__init__(name=name)
        self.loss = loss

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
