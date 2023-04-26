import setGPU
import hydra
from dataset import PhlatnetDataGenerator


from models.flatnet.discriminator import *
from models.flatnet.camera_inversion import *
from models.flatnet.gan import *

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




@hydra.main(version_base=None, config_path="configs", config_name="flatnet_reconstruction")
def main(config):
    # conf = tf.compat.v1.ConfigProto()
    # conf.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=conf)
    # sess.as_default()


    # physical_devices = tf.config.list_physical_devices('GPU')
    # try:
    #     for device in physical_devices:
    #         tf.config.experimental.set_memory_growth(device, True)
    #     print('allow memory grow')
    # except:
    #     # Invalid device or cannot modify virtual devices once initialized.
    #     pass

    now = datetime.now()

    dataset_config = config['dataset']

    indexes = np.arange(dataset_config['len'])

    train_indexes, val_indexes = train_test_split(indexes, 
                                                  test_size=config['validation_split'], 
                                                  shuffle=True,
                                                  random_state=config['seed'])
    
    # Data Generators
    train_generator = PhlatnetDataGenerator(dataset_config=dataset_config, 
                                           indexes=train_indexes, 
                                           batch_size=config['batch_size'], 
                                           seed=config['seed'])
    val_generator = PhlatnetDataGenerator(dataset_config=dataset_config,  
                                         indexes=val_indexes, 
                                         batch_size=config['batch_size'], 
                                         seed=config['seed'])


    lpips_loss = get_lpips_loss(config, 'vgg')

    # alpha_lpips = K.variable(1.0)
    # alpha_mse = K.variable(1.0)


    
    ### SIMPLE FLATNET without discriminator ###
    # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-02)
    # input_shape = get_shape(dataset_config, measure=True) 
    # model = get_inversion_model(config, input_shape)
    # mse_loss = MeanSquaredError()
    # loss = LossCombiner([lpips_loss, mse_loss], [alpha_lpips, alpha_mse], name='loss_combination')
    # model.compile(optimizer = optimizer, 
    #               loss = mse_loss,
    #               metrics = [MeanSquaredError(name='mse')])


    ### GAN FLATNET ####
    discriminator = get_discriminator(get_shape(dataset_config, measure=False))

    input_shape = get_shape(dataset_config, measure=True)  
    output_shape = get_shape(dataset_config, measure=False)                 

    inversion_model = get_inversion_model(config, input_shape)

    model = FlatNetGAN(discriminator, inversion_model)

    # TODO : put flatnet args instead = lpips:1.6, mse=1
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-03)
    model.compile(d_optimizer=d_optimizer,
                  g_optimizer=g_optimizer,
                  g_perceptual_loss=lpips_loss,
                  adv_weight=0.6,
                  mse_weight=1,
                  perc_weight=1.2,
                  metrics=[MeanSquaredError(name='mse'), 
                            lpips_loss, 
                            LossCombiner([lpips_loss, MeanSquaredError(name='mse')], [1, 1], name='total')])
    
    model.build(Input(shape=input_shape).shape)
        
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

    reduce_lr = ReduceLROnPlateau(monitor='val_total', factor=0.5, patience=3, min_lr=6e-08, verbose=1)
    
    callbacks = [model_checkpoint, reduce_lr]

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




