import setGPU
import hydra
from dataset import WallerlabGenerator
from models.unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split

import json
from utils import *

import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_first')
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler

from keras.losses import MeanSquaredError

from torchsummary import summary

from functools import partial
from datetime import datetime




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
    train_generator = WallerlabGenerator(dataset_config=dataset_config, 
                                         indexes=train_indexes,
                                         batch_size=config['batch_size'], 
                                         greyscale=config['greyscale'],
                                         use_crop=config['use_crop'], 
                                         seed=config['seed'])
    val_generator = WallerlabGenerator(dataset_config,  
                                       val_indexes, 
                                       batch_size=config['batch_size'],
                                       greyscale=config['greyscale'],
                    use_crop=config['use_crop'],  
                                       seed=config['seed'])

    # train_generator = shared_mem_multiprocessing(train_generator, workers=config['workers'], queue_max_size=config['queue_max_size'])
    # val_generator = shared_mem_multiprocessing(val_generator, workers=config['workers'], queue_max_size=config['queue_max_size'])

    # losses
    losses = []
    weights = []
    dynamic_weights = []
    for loss_id in config['loss'].keys():
        loss_config = config['loss'][loss_id]
        losses.append(get_loss_from_name(loss_id, loss_config, config))
        weight = loss_config['weight']
        # add to list to be updated during callback
        if loss_config['additive_factor']:
            weight = K.variable(weight)
            dynamic_weights.append((weight, loss_config['additive_factor']))
        weights.append(weight)
  
    loss = LossCombiner(losses, weights, name='loss_combination')

    # metrics
    metrics = []
    metric_weights = []
    for metric_id in config['metric'].keys():

        metric_config = config['metric'][metric_id]

        metrics.append(get_loss_from_name(metric_id, metric_config, config))
        metric_weights.append(metric_config['weight'])


    optimizer = tf.keras.optimizers.get(**config['optimizer'])

    in_shape = get_shape(dataset_config, measure=True, greyscale=config['greyscale'])
    out_shape = get_shape(dataset_config, measure=False, greyscale=config['greyscale'])

    model = u_net(in_shape, **config['model'], out_shape=out_shape)
    

    # TODO : define best fixed loss weighting for validation // flatnet = lpips:1.6, mse=1
    model.compile(optimizer = optimizer, 
                  loss = loss,
                  metrics = [*metrics,
                             LossCombiner(metrics, metric_weights, name='total')])

    print(model.summary())

    if not os.path.isdir(config['temp_store_path']):
        os.makedirs(config['temp_store_path'])
    checkpoint_path = os.path.join(config['temp_store_path'], 'checkpoint_' + str(now.strftime("%H-%M-%S")))
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=checkpoint_path,
                                            save_weights_only=True,
                                            monitor='val_total',
                                            mode='min',
                                            save_best_only=True,
                                            save_freq="epoch",
                                            verbose=1)
    
    
    callbacks = [ChangeLossWeights(dynamic_weights), model_checkpoint]

    if config['lr_reducer']['type']:
        if config['lr_reducer']['type'] == "reduce_lr_on_plateau":
            callbacks.append(ReduceLROnPlateau(**config['lr_reducer']['reduce_lr_on_plateau']))
        elif config['lr_reducer']['type'] == "learning_rate_scheduler":
            callbacks.append(LearningRateScheduler(**config['lr_reducer']['learning_rate_scheduler']))
        else:
            raise ValueError(config['lr_reducer']['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")



    if config['use_tensorboard']:
        tb_path = os.path.join(config['tensorboard_path'], str(now.date()), str(now.strftime("%H-%M-%S")), 'logs')
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)
        tb_callback = tf.keras.callbacks.TensorBoard(tb_path, 
                                                     histogram_freq = 1, 
                                                     update_freq='epoch')
        callbacks.append(tb_callback)

    

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
        # name =  config['model']['name'] + "_" + str(now.strftime("%H-%M-%S")) + '.pb'
        name =  config['model']['name'] + ".pb"
        tf.saved_model.save(model, os.path.join(store_path, name))


    


if __name__ == "__main__":
    main()
    print('done')




