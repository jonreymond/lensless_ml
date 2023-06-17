import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import setGPU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'
# from utils import *
import hydra

# from dataset import get_dataset
from models.unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from models.model_utils import *

import json


import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_last')
from keras import backend as K


from keras.losses import MeanSquaredError


from functools import partial
from datetime import datetime

from tf_dataset import *


from hydra.utils import get_original_cwd, to_absolute_path

import tensorflow_model_optimization as tfmot
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

    # if config['debug_mode']:
    #     tf.debugging.experimental.enable_dump_debug_info(
    #     os.path.join(store_folder, 'tensorboard_logs'), tensor_debug_mode="FULL_HEALTH")


    # tf.debugging.set_log_device_placement(True)

    gpus = tf.config.list_logical_devices('GPU')
    # communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.RING)

    # strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
    # # #strategy = tf.distribute.MirroredStrategy(gpus)#, cross_device_ops=tf.distribute.ReductionToOneDevice())#tf.distribute.HierarchicalCopyAllReduce())
    # with strategy.scope():
    for i in range(1):
        
        # with open(os.path.join(store_folder, 'output.txt'), 'w') as f:
        #     sys.stdout = Tee(sys.stdout, f)
        #     sys.stderr = Tee(sys.stderr, f)


        # for j in range(1):
            print('number of gpus: ', len(gpus))
            local_batch_size = config['batch_size'] #// len(gpus)



        # for i in range(1):
        

            # print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))

            # gpus = tf.config.list_physical_devices('GPU')
            # tf.config.set_visible_devices(gpus[0], 'GPU')


            dataset_config = config['dataset']
            indexes = np.arange(dataset_config['len'])
            if config['test']:
                indexes = indexes[:80]

            train_indexes, val_indexes = train_test_split(indexes, 
                                                        test_size=config['validation_split'], 
                                                        shuffle=True,
                                                        random_state=config['seed'])
            
            # Data Generators

            resize_input_shape = None
            if config['resize_input'] and config['dataset']['name'] not in['flatnet', 'phlatnet']:
                raise NotImplementedError('Resize input is not implemented for this dataset')

            data_args = dict(batch_size=local_batch_size,#config['batch_size'], 
                            greyscale=config['greyscale'],
                            use_crop=config['use_crop'], 
                            seed=config['seed'])
            
            if config['resize_input']:
                resize_input_shape = (config['resize_input_height'], config['resize_input_width'])
                data_args['input_shape'] = resize_input_shape
            

            print('train length: ',len(train_indexes), ', val length: ', len(val_indexes))
            assert len(train_indexes) > 0 and len(val_indexes) > 0 and len(train_indexes)> len(val_indexes), 'Wrong train/val split'
            
            # train = get_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args)
            # # train = tf.data.Dataset.from_generator(train)
            # val = get_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args)

            train = get_tf_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args).get()
            val = get_tf_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args).get()

            # val = strategy.experimental_distribute_dataset(val)
            # train = strategy.experimental_distribute_dataset(train)


            # for x, y in val.take(1):
            #     print('x shape: ', x.shape)
            #     print('y shape: ', y.shape)


            # train = shared_mem_multiprocessing(train, workers=config['workers'], queue_max_size=32)
            # val = shared_mem_multiprocessing(val, workers=config['workers'], queue_max_size=32)

            # train = tf.data.Dataset.from_generator(train)
            # val = tf.data.Dataset.from_generator(train)

            input_shape = get_shape(dataset_config, measure=True, greyscale=config['greyscale'], resize_input_shape=resize_input_shape)

            output_shape = get_shape(dataset_config, measure=False, greyscale=config['greyscale'], resize_input_shape=resize_input_shape)

            # losses
            loss_dict, dynamic_weights = get_losses(config, output_shape, local_batch_size)
            # metrics
            metrics, metric_weights = get_metrics(config, output_shape, local_batch_size)


            opt_config = dict(config['optimizer'])
            Opt_class = tf.keras.optimizers.get(opt_config['identifier']).__class__
            opt_config.pop('identifier')
            optimizer = Opt_class(**opt_config)
            # don't work with tf 12.0
            # optimizer = tf.keras.optimizers.get(config['optimizer']['identifier'], **config['optimizer']['kwargs'])         


            input = Input(shape=input_shape, name='input', dtype='float32')
            # camera inversion layer #
            camera_inversion_layer = None
            if config['use_camera_inversion']:
                # Separable dataset
                if dataset_config['type_mask'] == 'separable':
                    if config['random_init']:
                        raise NotImplementedError('Random init not implemented for separable dataset')
                    else:
                        assert dataset_config['name'] == 'flatnet', 'Only flatnet dataset is supported for separable dataset'
                        phi_l, phi_r = get_separable_init_matrices(dataset_config)
                    camera_inversion_layer = SeparableLayer(phi_l, phi_r)
                # Non separable dataset
                else:
                    if config['random_init']:
                        raise NotImplementedError('Random init not implemented for non separable dataset')
                    else:
                        psf = get_psf(dataset_config, input_shape=resize_input_shape)
                    camera_inversion_layer = FTLayer(psf=psf, **config['camera_inversion_args']['non_separable'])


            if camera_inversion_layer:
                input = camera_inversion_layer(input)
                
            model_config = dict(config['model'][config['model_name']])
            model_type = model_config.pop('type')
            # TODO : add experimental support for other models
            if model_type == 'unet':
                model_output = [u_net(input=input, **model_config, out_shape=output_shape)]
            else:
                model_output = [experimental_models(model_name=config['model_name'], 
                                                    input=input, 
                                                    out_shape=output_shape,
                                                    model_args=model_config['args'])]

            perceptual_model = Model(inputs=[input],
                                     outputs=model_output,
                                     name='perceptual_model')
            
            gen_model = ReconstructionModel3(input_shape=input_shape, 
                                            # output_shape=output_shape, 
                                            camera_inversion_layer=camera_inversion_layer,
                                            perceptual_model=perceptual_model)      


            discr_args = None
            if config['use_discriminator']:
                d_model = get_discriminator(output_shape, **config['discriminator']['model'])

                d_opt_config = dict(config['discriminator']['optimizer']['args'])
                D_opt_class = tf.keras.optimizers.get(d_opt_config['identifier']).__class__
                d_opt_config.pop('identifier')
                print(d_opt_config)
                d_optimizer = D_opt_class(**d_opt_config)

                adv_weight = config['discriminator']['weight']
                discr_args = dict(model=d_model, optimizer=d_optimizer, adv_weight=adv_weight, label_smoothing=config['discriminator']['label_smoothing'])



            model = compile_model(gen_model=gen_model, 
                                gen_optimizer=optimizer, 
                                loss_dict=loss_dict, 
                                metrics=metrics, 
                                metric_weights=metric_weights, 
                                discr_args=discr_args,
                                in_shape=input_shape, 
                                out_shape=output_shape,
                                global_batch_size=local_batch_size,#config['batch_size'],
                                distributed_gpu=config['distributed_gpu'],
                                num_gpus=len(gpus))
            
            
            
            assert int(config['weight_pruning']) + int(config['weight_clustering']) + int(config['QAT']) <= 1, 'At most one weight modification can be applied at a time'

            if config['weight_pruning']:
                prune_camera_inversion_layer = tfmot.sparsity.keras.prune_low_magnitude(gen_model.camera_inversion_layer)
                prune_perceptual_model = tfmot.sparsity.keras.prune_low_magnitude(gen_model.perceptual_model)
                gen_model = ReconstructionModel3(input_shape=input_shape,
                                                camera_inversion_layer=prune_camera_inversion_layer,
                                                perceptual_model=prune_perceptual_model)
            

            if config['weight_clustering']:
                cluster_camera_inversion_layer = tfmot.clustering.keras.cluster_weights(gen_model.camera_inversion_layer,
                                                                                        **config['clustering_params'])
                cluster_perceptual_model = tfmot.clustering.keras.cluster_weights(gen_model.perceptual_model,
                                                                                        **config['clustering_params'])
                gen_model = ReconstructionModel3(input_shape=input_shape,
                                                camera_inversion_layer=cluster_camera_inversion_layer,
                                                perceptual_model=cluster_perceptual_model)

            
            if config['QAT']:
                qat_camera_inversion_layer = quantize_annotate_layer(gen_model.camera_inversion_layer, InversionLayerQuantizeConfig())
                qat_perceptual_model = quantize_annotate_model(gen_model.perceptual_model)

                # gen_model = ReconstructionModel3(input_shape=input_shape, 
                #                         # output_shape=output_shape, 
                #                             camera_inversion_layer=qat_camera_inversion_layer,
                #                             perceptual_model=qat_perceptual_model)

                qat_camera_inversion_layer = tf.keras.Sequential([Input(shape=input_shape, name='input', dtype='float32'),
                                                qat_camera_inversion_layer])
                
                gen_model = qat_perceptual_model



                with quantize_scope({'InversionLayerQuantizeConfig': InversionLayerQuantizeConfig,
                                     'FTLayer': FTLayer}):
                    scheme = dict(QAT=tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme(),
                                  CQAT=tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(),
                                  PQAT=tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme(),
                                  PCQAT=tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))
                    qat_perceptual_model = quantize_apply(gen_model, scheme=scheme[config['QAT_scheme']])
                    qat_camera_inversion_layer = quantize_apply(qat_camera_inversion_layer, scheme=scheme[config['QAT_scheme']])

                    # print(qat_camera_inversion_layer.summary())
                    # print(qat_perceptual_model.summary())

                    gen_model = ReconstructionModel3(input_shape=input_shape,
                                                    camera_inversion_layer=qat_camera_inversion_layer,
                                                    perceptual_model=qat_perceptual_model)
                    print(gen_model.summary())
                    sys.exit()
                    


            model = compile_model(gen_model=gen_model, 
                                gen_optimizer=optimizer, 
                                loss_dict=loss_dict, 
                                metrics=metrics, 
                                metric_weights=metric_weights, 
                                discr_args=discr_args,
                                in_shape=input_shape, 
                                out_shape=output_shape,
                                global_batch_size=local_batch_size,#config['batch_size'],
                                distributed_gpu=config['distributed_gpu'],
                                num_gpus=len(gpus))
            
            print('='*70)
            print('='*70)
            print('='*70)
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
                    verbose=config['verbose'])

            model.load_weights(checkpoint_path).expect_partial()

            if config['save']:
                print('saving model ...')
                store_path = os.path.join(store_folder, 'models')
                if not os.path.isdir(store_path):
                    os.makedirs(store_path)
                

                # save generator model
                # name_gen = 'gen_' + config['model_name'] + '_.pb'
                # tf.saved_model.save(gen_model, os.path.join(store_path, name_gen))

                # 
                name_gen = 'gen_' + config['model_name']
                if config['weight_pruning']:
                    name_gen += '_pruned'
                    print('size before pruning: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')
                    gen_model = tfmot.sparsity.keras.strip_pruning(gen_model)
                    print('size after pruning: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')
                    
                if config['weight_clustering']:
                    name_gen += '_clustered'
                    print('size before clustering: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')
                    gen_model = tfmot.clustering.keras.strip_clustering(gen_model)
                    print('size after clustering: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')

                name_gen += '.pb'

                tf.keras.models.save_model(gen_model, os.path.join(store_path, name_gen), include_optimizer=False)

                if gen_model.camera_inversion_layer:
                    # save weights of camera inversion layer in numpy format
                    # if not os.path.isdir(os.path.join(store_path, 'camera_inversion')):
                    #     os.makedirs(os.path.join(store_path, 'camera_inversion'))
                    # for weight in gen_model.camera_inversion_layer.get_list_weights():
                    #     np.save(os.path.join(store_path, 'camera_inversion', weight.name + '.npy'), weight.numpy())
                    camera_inversion = gen_model.camera_inversion_layer
                    if not isinstance(camera_inversion, tf.keras.Sequential):
                        camera_inversion = tf.keras.Sequential([Input(shape=input_shape, name='input', dtype='float32'),
                                                                camera_inversion])
                    tf.keras.models.save_model(camera_inversion, os.path.join(store_path, 'camera_inversion.pb'), include_optimizer=False)
                
                # save perceptual model
                tf.keras.models.save_model(gen_model.perceptual_model, os.path.join(store_path, 'perceptual_model.pb'), include_optimizer=False)




                name =  config['model_name'] + '.pb'
                tf.saved_model.save(model, os.path.join(store_path, name))



            sys.stdout = sys.__stdout__



    


if __name__ == "__main__":
    main()
    print('done')




