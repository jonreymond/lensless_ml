import os
import tensorflow as tf
from lpips import LPIPS
import torch
from torch_to_tf import to_tf_graph
from keras import backend as K
import sys



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
        torch_shape = [data_spec['shape'][-1], *data_spec['shape'][:-1]]
        sample_input = (torch.randn(dataset_config['batch_size'], *torch_shape, requires_grad=False).cuda(),
                        torch.randn(dataset_config['batch_size'], *torch_shape, requires_grad=False).cuda())
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    stored_lpips = tf.keras.models.load_model(lpips_path + '.pb')
    
    lpips = lambda x, y : tf.reduce_mean(stored_lpips(input1=tf.transpose(x, perm=[0, 3, 1, 2]), input2=tf.transpose(y, perm=[0, 3, 1, 2]))['output'])
    return lpips



class ChangeLossWeights(tf.keras.callbacks.Callback):
    def __init__(self, alpha_minus, alpha_plus, multiplier):
        self.alpha_minus =alpha_minus
        self.alpha_plus = alpha_plus
        self.mult = multiplier

    def on_epoch_end(self, epoch, logs=None):
        K.set_value(self.alpha_minus, self.alpha_minus * self.mult)
        K.set_value(self.alpha_plus, self.alpha_plus / self.mult)


def weighted_loss(target, output, loss_function, alpha):
    # resample
    return loss_function(target, output) * alpha

class LpipsCallback(tf.keras.callbacks.Callback):
    def __init__(self, lpips_loss, val_generator):
        self.lpips_loss = lpips_loss
        self.val_generator = val_generator

    def on_epoch_end(self, epoch, logs=None):
        res = []
        print(' lpips computation...')
        for val_data in self.val_generator:

            x, y = val_data
            x, y = tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(y, dtype=tf.float32)
            res.append(self.lpips_loss(x, y))

        # sys.stdout.write('\r')
        print(' \r lpips score :', (sum(res) / len(res)).numpy())