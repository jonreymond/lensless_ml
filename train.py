import setGPU
import hydra
from dataset import DataGenerator
from unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

import json
from utils import *

import tensorflow as tf
from keras import backend as K

from tensorflow.keras.losses import MeanSquaredError

from torchsummary import summary
from functools import partial



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

    lpips_loss = get_lpips_loss(config, data_spec)


    alpha_lpips = K.variable(0.5)
    alpha_mse = K.variable(1)
    
    lpips_weighted = partial(weighted_loss, loss_function=lpips_loss, alpha=alpha_lpips)
    mse_weighted = partial(weighted_loss, loss_function=MeanSquaredError(), alpha=alpha_mse)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)

    model = u_net(data_spec['shape'])


    model.compile(loss = [mse_weighted, lpips_weighted], 
                  optimizer = optimizer, 
                  metrics = [MeanSquaredError()])

    print(model.summary())

    callbacks = [ChangeLossWeights(alpha_lpips, alpha_mse, 0.99)]

    model.fit(train_generator,
            epochs=10,
            callbacks=callbacks,
            validation_data=val_generator,
            use_multiprocessing=True,
            shuffle=False)



    


if __name__ == "__main__":
    main()
    print('done')




