import hydra
from dataset import DataGenerator
from unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from lpips import LPIPS
import json
import torch
from torch_to_tf import to_tf_graph
import tensorflow as tf
from keras import backend as K

from tensorflow.keras.losses import MeanSquaredError

from torchsummary import summary
from functools import partial


def get_lpips_name(config, data_spec):
    shape_str = ''
    for s in data_spec['shape']:
        shape_str += '_' + str(s)
    return 'lpips_' + config['train_params']['lpips_model'] + '_shape' + shape_str


class ChangeLossWeights(tf.keras.callbacks.Callback):
    def __init__(self, alpha1, alpha2, multiplier):
        self.alpha1 =alpha1
        self.alpha2 = alpha2
        self.mult = multiplier

    def on_epoch_end(self, epoch, logs=None):
        K.set_value(self.alpha1, self.alpha1 * self.mult)
        K.set_value(self.alpha2, self.alpha2 / self.mult)


def weighted_loss(target, output, loss_function, alpha):
    # resample
    return loss_function(target, output) * alpha



@hydra.main(version_base=None, config_path="", config_name="ml_reconstruction")
def main(config):
    

    dataset_config = config['dataset']

    spec_json_name = [name for name in os.listdir(dataset_config['path']) if '.json' in name][0]
    data_spec = json.load(open(os.path.join(dataset_config['path'], spec_json_name)))

    indexes = np.arange(data_spec['len'])

    train_indexes, val_indexes = train_test_split(indexes, 
                                                  test_size=dataset_config['validation_split'], 
                                                  shuffle=True,
                                                  random_state=config['seed'])
    
    # Data Generators
    train_generator = DataGenerator(dataset_config, data_spec, train_indexes, config['seed'])
    val_generator = DataGenerator(dataset_config, data_spec, val_indexes, config['seed'])

    # Don't work: Torch
    train_config = config['train_params']
    

    if not os.path.isdir('lpips_losses'):
        os.makedirs('lpips_losses')

    lpips_path = os.path.join('lpips_losses', get_lpips_name(config, data_spec))

    if not os.path.isfile(lpips_path + '.pb'):
        lpips_loss = LPIPS(net=train_config ['lpips_model']).to(torch.device('cuda:0'))
        #change to satisfy with torch order : first channels
        torch_shape = [data_spec['shape'][-1], *data_spec['shape'][:-1]]
        sample_input = (torch.randn(dataset_config['batch_size'], *torch_shape, requires_grad=False).to(torch.device('cuda:0')),
                        torch.randn(dataset_config['batch_size'], *torch_shape, requires_grad=False).to(torch.device('cuda:0')))
        to_tf_graph(lpips_loss, sample_input, lpips_path)

    lpips_loss = tf.keras.models.load_model(lpips_path + '.pb')


    # model = Model(inputs = input, outputs = [y1,y2])
    alpha_lpips = K.variable(0.5)
    alpha_mse = K.variable(1)
    
    lpips_weighted = partial(weighted_loss, loss_function=lpips_loss, alpha=alpha_lpips)
    mse_weighted = partial(weighted_loss, loss_function=MeanSquaredError(), alpha=alpha_mse)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)

    model = u_net(data_spec['shape'])

    model.compile(loss = [lpips_weighted, mse_weighted], 
                  optimizer = optimizer, 
                  metrics = [lpips_loss, MeanSquaredError()])

    print(model.summary())


    


if __name__ == "__main__":
    main()
    print('done')




