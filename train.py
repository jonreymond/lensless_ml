import setGPU
import hydra
from dataset import get_dataset
from models.unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from models.model_utils import *

import json
from utils import *

import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_first')
from keras import backend as K


from keras.losses import MeanSquaredError

from torchsummary import summary

from functools import partial
from datetime import datetime

from tf_dataset import *






@hydra.main(version_base=None, config_path="configs", config_name="wallerlab_reconstruction")
def main(config):
    
    now = datetime.now()

    dataset_config = config['dataset']
    indexes = np.arange(dataset_config['len'])

    train_indexes, val_indexes = train_test_split(indexes, 
                                                  test_size=config['validation_split'], 
                                                  shuffle=True,
                                                  random_state=config['seed'])
    
    # Data Generators
    data_args = dict(batch_size=config['batch_size'], 
                    greyscale=config['greyscale'],
                    use_crop=config['use_crop'], 
                    seed=config['seed'])
    
    train_generator = get_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args)
    val_generator = get_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args)

    # train_generator = get_tf_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args)
    # val_generator = get_tf_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args)


    # train_generator = shared_mem_multiprocessing(train_generator, workers=config['workers'], queue_max_size=config['queue_max_size'])
    # val_generator = shared_mem_multiprocessing(val_generator, workers=config['workers'], queue_max_size=config['queue_max_size'])

    

    # losses
    loss_dict, dynamic_weights = get_losses(config)

    # metrics
    metrics, metric_weights = get_metrics(config)


    optimizer = tf.keras.optimizers.get(**config['optimizer'])

    
    in_shape = get_shape(dataset_config, measure=True, greyscale=config['greyscale'])
    out_shape = get_shape(dataset_config, measure=False, greyscale=config['greyscale'])


    model = get_model(config=config, input_shape=in_shape, out_shape=out_shape, model_name='wallerlab_model')


    model = compile_model(model=model, 
                          optimizer=optimizer, 
                          loss_dict=loss_dict, 
                          metrics=metrics, 
                          metric_weights=metric_weights, 
                          config=config, 
                          in_shape=in_shape, 
                          out_shape=out_shape)

    print(model.summary())


    if not os.path.isdir(config['temp_store_path']):
        os.makedirs(config['temp_store_path'])
    checkpoint_path = os.path.join(config['temp_store_path'], 'checkpoint_' + str(now.strftime("%H-%M-%S")))

    callbacks = get_callbacks(model=model, 
                              checkpoint_path=checkpoint_path, 
                              config=config, 
                              dynamic_weights=dynamic_weights,
                              now=now)
    
    model.fit(train_generator,
            epochs=config['epochs'],
            callbacks=callbacks,
            validation_data=val_generator,
            use_multiprocessing=True,
            workers=config['workers'],
            shuffle=True,
            verbose=1) # check if need = False

    model.load_weights(checkpoint_path).expect_partial()

    if config['save']:
        print('saving model ...')
        store_path = config['model_path'] + '/' + str(now.date())
        if not os.path.isdir(store_path):
            os.makedirs(store_path)
        name =  config['model']['name'] + "_" + str(now.strftime("%H-%M-%S")) + '.pb'
        # name =  config['model']['name'] + ".pb"
        tf.saved_model.save(model, os.path.join(store_path, name))


    


if __name__ == "__main__":
    main()
    print('done')




