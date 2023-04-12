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
from keras.callbacks import ReduceLROnPlateau

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
                                         seed=config['seed'])
    val_generator = WallerlabGenerator(dataset_config,  
                                       val_indexes, 
                                       batch_size=config['batch_size'],
                                       greyscale=config['greyscale'], 
                                       seed=config['seed'])


    lpips_loss = get_lpips_loss(config)


    alpha_lpips = K.variable(1.0)
    alpha_mse = K.variable(1.0)

    mse_loss = MeanSquaredError()

    loss = LossCombiner([lpips_loss, mse_loss], [alpha_lpips, alpha_mse], name='loss_combination')

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-02)

    model = u_net(get_shape(dataset_config, measure=True, greyscale=config['greyscale']), **config['model'])
    

    # TODO : define best fixed loss weighting for validation // flatnet = lpips:1.6, mse=1
    model.compile(optimizer = optimizer, 
                  loss = loss,
                  metrics = [MeanSquaredError(name='mse'), 
                             lpips_loss, 
                             LossCombiner([lpips_loss, mse_loss], [1, 1], name='total')])

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
    
    reduce_lr = ReduceLROnPlateau(monitor='val_total', factor=0.1, patience=3, min_lr=6e-08, verbose=1)
    
    callbacks = [ChangeLossWeights(alpha_plus=alpha_lpips, alpha_minus=alpha_mse, factor=0.1), model_checkpoint, reduce_lr]

    if config['use_tensorboard']:
        tb_callback = tf.keras.callbacks.TensorBoard(os.path.join(config['temp_store_path'], 'logs'), 
                                                     histogram_freq = 1, 
                                                     update_freq='epoch')
        callbacks.append(tb_callback)

    

    model.fit(train_generator,
            epochs=config['epochs'],
            callbacks=callbacks,
            validation_data=val_generator,
            use_multiprocessing=True,
            shuffle=True) # check if need = False

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




