# import setGPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from utils import *
import hydra

from dataset import get_dataset
from models.unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from models.model_utils import *

import json


import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_first')
from keras import backend as K


from keras.losses import MeanSquaredError


from functools import partial
from datetime import datetime

from tf_dataset import *

import tensorflow_model_optimization as tfmot

from hydra.utils import get_original_cwd, to_absolute_path

from tensorflow_model_optimization.quantization.keras import quantize_scope, quantize_annotate_layer, quantize_apply, quantize_annotate_model







@hydra.main(version_base=None, config_path="configs", config_name="wallerlab_reconstruction")
def main(config):
    # tf.config.run_functions_eagerly(False)


    print('='*80)
    print('='*80)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    store_folder = hydra_cfg['runtime']['output_dir']
    print('Store folder: ', store_folder)
    print('-'*80)

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    # # Create 2 virtual GPUs with 1GB memory each
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
    #             tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)
    # tf.debugging.set_log_device_placement(True)
    # gpus = tf.config.list_logical_devices('GPU')
    # strategy = tf.distribute.MirroredStrategy(gpus)
    # with strategy.scope():
    for i in range(1):
        for j in range(1):


        # copy the config file in the store folder
        # f = open(os.path.join(store_folder, 'config.yaml'), 'w')
        # sys.stdout = Tee(sys.stdout, f)
        # sys.stderr = Tee(sys.stderr, f)
        # for i in range(1):
        # with open(os.path.join(store_folder, 'output.txt'), 'w') as f:
        #     sys.stdout = Tee(sys.stdout, f)
        #     sys.stderr = Tee(sys.stderr, f)


            # print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

            # gpus = tf.config.list_physical_devices('GPU')
            # tf.config.set_visible_devices(gpus[0], 'GPU')


            # sys.exit()


            dataset_config = config['dataset']
            indexes = np.arange(dataset_config['len'])
            if config['test']:
                indexes = indexes[:1000]

            train_indexes, val_indexes = train_test_split(indexes, 
                                                        test_size=config['validation_split'], 
                                                        shuffle=True,
                                                        random_state=config['seed'])
            
            # Data Generators
            data_args = dict(batch_size=config['batch_size'], 
                            greyscale=config['greyscale'],
                            use_crop=config['use_crop'], 
                            seed=config['seed'])
            

            print('train length: ',len(train_indexes), ', val length: ', len(val_indexes))
            assert len(train_indexes) > 0 and len(val_indexes) > 0 and len(train_indexes)> len(val_indexes), 'Wrong train/val split'
            
            train = get_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args)
            # train = tf.data.Dataset.from_generator(train)
            val = get_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args)



            # train = get_tf_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args)
            # val = get_tf_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args)


            # train = shared_mem_multiprocessing(train, workers=config['workers'], queue_max_size=32)
            # val = shared_mem_multiprocessing(val, workers=config['workers'], queue_max_size=32)

            # train = tf.data.Dataset.from_generator(train)
            # val = tf.data.Dataset.from_generator(train)


            # losses
            loss_dict, dynamic_weights = get_losses(config)

            # metrics
            metrics, metric_weights = get_metrics(config)


            opt_config = dict(config['optimizer'])
            Opt_class = tf.keras.optimizers.get(opt_config['identifier']).__class__
            opt_config.pop('identifier')
            optimizer = Opt_class(**opt_config)
            # don't work with tf 12.0
            # optimizer = tf.keras.optimizers.get(config['optimizer']['identifier'], **config['optimizer']['kwargs'])

            
            input_shape = get_shape(dataset_config, measure=True, greyscale=config['greyscale'])
            output_shape = get_shape(dataset_config, measure=False, greyscale=config['greyscale'])


            # gen_model = get_model(config=config, input_shape=in_shape, out_shape=out_shape, model_name='the_model')

            model_config = dict(config['model'][config['model_name']])
            camera_inversion_args = config if config['use_camera_inversion'] else None
            model_config.pop('type')

            gen_model = ReconstructionModel(input_shape = input_shape, 
                                            output_shape = output_shape, 
                                            unet_args = model_config, 
                                            camera_inversion_args = camera_inversion_args,
                                            # model_name='reconstruction model'
                                            )
            

            gen_model.build((None, *input_shape))


            print(gen_model.summary())
            

            
            if config['use_QAT']:
                # model = tfmot.quantization.keras.quantize_annotate_model(
                #     tf.keras.Sequential([
                # tfmot.quantization.keras.quantize_annotate_layer(gen_model.camera_inversion_layer, DefaultDenseQuantizeConfig())
                # ]))

                # # `quantize_apply` requires mentioning `DefaultDenseQuantizeConfig` with `quantize_scope`
                # # as well as the custom Keras layer.
                # with tfmot.quantization.keras.quantize_scope(
                # {'DefaultDenseQuantizeConfig': DefaultDenseQuantizeConfig,
                # 'CustomLayer': CustomLayer}):
                # # Use `quantize_apply` to actually make the model quantization aware.
                # quant_aware_model = tfmot.quantization.keras.quantize_apply(model)
                

                
                # gen_model = tfmot.quantization.keras.quantize_model(gen_model)
                # a = quantize_annotate_layer(gen_model.camera_inversion_layer)

                model = tf.keras.Sequential([quantize_annotate_layer(gen_model.camera_inversion_layer, FTLayerQuantizeConfig()),
                                             quantize_annotate_model(gen_model.perceptual_model)])
                model.build((None, *input_shape))
                with quantize_scope({'FTLayerQuantizeConfig': FTLayerQuantizeConfig,
                                     'FTLayer': FTLayer}):
                    model = quantize_apply(model)
                gen_model = model

                print('done')


            discr_args = None
            if config['use_discriminator']:
                d_model = get_discriminator(output_shape, **config['discriminator']['model'])

                d_opt_config = dict(config['discriminator']['optimizer']['args'])
                D_opt_class = tf.keras.optimizers.get(d_opt_config['identifier']).__class__
                d_opt_config.pop('identifier')
                print(d_opt_config)
                d_optimizer = D_opt_class(**d_opt_config)
                print(d_model.summary())
                adv_weight = config['discriminator']['weight']
                discr_args = dict(model=d_model, optimizer=d_optimizer, adv_weight=adv_weight)




            model = compile_model(gen_model=gen_model, 
                                gen_optimizer=optimizer, 
                                loss_dict=loss_dict, 
                                metrics=metrics, 
                                metric_weights=metric_weights, 
                                discr_args=discr_args,
                                in_shape=input_shape, 
                                out_shape=output_shape)

            print(model.summary())

            


            checkpoint_path = os.path.join(store_folder, 'checkpoints')
            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_path += '/'

            callbacks = get_callbacks(model=model, 
                                    store_folder=store_folder,
                                    checkpoint_path=checkpoint_path, 
                                    config=config, 
                                    dynamic_weights=dynamic_weights)
            

            if config['load_pretrained']:
                print('loading pretrained model ...')
                model.load_weights(config['pretrained_path']).expect_partial()
                print(model.optimizer.get_config())

            
            model.fit(train,
                    epochs=config['epochs'],
                    callbacks=callbacks,
                    validation_data=val,
                    use_multiprocessing=True,
                    workers=config['workers'],
                    shuffle=True,
                    verbose=1) # check if need = False

            model.load_weights(checkpoint_path).expect_partial()

            if config['save']:
                print('saving model ...')
                store_path = os.path.join(store_folder, 'models')
                if not os.path.isdir(store_path):
                    os.makedirs(store_path)
                name =  config['model_name'] + '.pb'

                tf.saved_model.save(model, os.path.join(store_path, name))

                name_gen = 'gen_' + config['model_name'] + '_.pb'
                tf.saved_model.save(gen_model, os.path.join(store_path, name_gen))


            sys.stdout = sys.__stdout__



    


if __name__ == "__main__":
    main()
    print('done')




