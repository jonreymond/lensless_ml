# #############################################################################
# train.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import setGPU

import hydra


import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split


from model import *
from callbacks import get_callbacks
from utils import *



import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_last')
from keras import backend as K

from tf_dataset import *


from hydra.utils import get_original_cwd, to_absolute_path

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.quantization.keras import quantize_scope, quantize_annotate_layer, quantize_apply, quantize_annotate_model





@hydra.main(version_base=None, config_path="configs", config_name="train_reconstruction")
def main(config):

    print('='*80)
    print('='*80)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    store_folder = hydra_cfg['runtime']['output_dir']
    print('Store folder: ', store_folder)
    print('-'*80)


    gpus = tf.config.list_logical_devices('GPU')

    for i in range(1):
        # To record terminal output, uncomment the following lines
        # with open(os.path.join(store_folder, 'output.txt'), 'w') as f:
        #     sys.stdout = Tee(sys.stdout, f)
        #     sys.stderr = Tee(sys.stderr, f)

            print('number of gpus: ', len(gpus))


            ################################################################################################################
            ###################################### Dataset generation ######################################################
            ################################################################################################################

            dataset_config = config['dataset']
            indexes = np.arange(dataset_config['len'])
            if config['test']:
                indexes = indexes[:80]

            train_indexes, val_indexes = train_test_split(indexes, 
                                                        test_size=config['validation_split'], 
                                                        shuffle=True,
                                                        random_state=config['seed'])
            
            resize_input_shape = None
            if config['resize_input'] and config['dataset']['name'] not in['flatnet', 'phlatnet']:
                raise NotImplementedError('Resize input is not implemented for this dataset')

            data_args = dict(batch_size=config['batch_size'],
                            greyscale=config['greyscale'],
                            use_crop=config['use_crop'], 
                            seed=config['seed'])
            
            if config['resize_input']:
                resize_input_shape = (config['resize_input_height'], config['resize_input_width'])
                data_args['input_shape'] = resize_input_shape
            

            print('train length: ',len(train_indexes), ', val length: ', len(val_indexes))
            assert len(train_indexes) > 0 and len(val_indexes) > 0 and len(train_indexes)> len(val_indexes), 'Wrong train/val split'
            

            train = get_tf_dataset(config['dataset']['name'], dataset_config, train_indexes, data_args).get()
            val = get_tf_dataset(config['dataset']['name'], dataset_config, val_indexes, data_args).get()


            ################################################################################################################
            ###################################### Metrics, losses and optimizer ###########################################
            ################################################################################################################


            input_shape = get_shape(dataset_config, measure=True, greyscale=config['greyscale'], resize_input_shape=resize_input_shape)
            output_shape = get_shape(dataset_config, measure=False, greyscale=config['greyscale'], resize_input_shape=resize_input_shape)


            loss_dict = dict()
            # if the weights change during training, we need to use K.variable
            dynamic_weights = []
            for loss_id in config['loss'].keys():
                loss_config = config['loss'][loss_id]

                weight = loss_config['weight']
                # add to list to be updated during callback
                if loss_config['additive_factor']:
                    weight = K.variable(weight)
                    dynamic_weights.append((weight, loss_config['additive_factor']))

                loss_args = None
                if loss_id == 'lpips':
                    loss_args = dict(shape=output_shape, batch_size=config['batch_size'], type_model=loss_config['type_model'])
                loss_dict.update({loss_id: (weight, get_loss(loss_id, config['distributed_gpu'], loss_args))})


            metrics = []
            metric_weights = []
            for metric_id in config['metric'].keys():
                metric_config = config['metric'][metric_id]
                metric_args = None
                if metric_id == 'lpips':
                    metric_args = dict(shape=output_shape, batch_size=config['batch_size'], type_model=metric_config['type_model'])
                metrics.append(get_metric(metric_id, config['distributed_gpu'], metric_args))
                metric_weights.append(metric_config['weight'])



            opt_config = dict(config['optimizer'])
            Opt_class = tf.keras.optimizers.get(opt_config['identifier']).__class__
            opt_config.pop('identifier')
            optimizer = Opt_class(**opt_config)
      

            ################################################################################################################
            ###################################### Model generation ########################################################
            ################################################################################################################

            psf, phi_l, phi_r = None, None, None

            if config['use_camera_inversion']:
                # Separable dataset
                if dataset_config['type_mask'] == 'separable':
                    if config['random_init']:
                        raise NotImplementedError('Random init not implemented for separable dataset')
                    else:
                        phi_l, phi_r = get_separable_init_matrices(dataset_config)
                else:
                    if config['random_init']:
                        raise NotImplementedError('Random init not implemented for non separable dataset')
                    else:
                        psf = get_psf(dataset_config, input_shape=resize_input_shape)

            camera_inversion_args = None
            if config['use_camera_inversion']:
                type_mask = dataset_config['type_mask']
                if type_mask == 'separable':
                    camera_inversion_args = dict(type=type_mask)
                elif type_mask == 'non_separable':
                    camera_inversion_args = config['camera_inversion_non_separable']
                                             
            gen_model = UnetModel(input_shape=input_shape,
                                    output_shape=output_shape,
                                    perceptual_args=config.model[config['model_name']],
                                    camera_inversion_args=camera_inversion_args,
                                    model_weights_path=config.model_weights_path,
                                    psf=psf,
                                    phi_l=phi_l,
                                    phi_r=phi_r,
                                    name=config.model_name)     


        ################################################################################################################
        ############################################# Discriminator setup ##############################################
        ################################################################################################################

            discr_args = None
            if config['use_discriminator']:

                d_model = Discriminator(output_shape, **config['discriminator']['model'])

                d_opt_config = dict(config['discriminator']['optimizer']['args'])
                D_opt_class = tf.keras.optimizers.get(d_opt_config['identifier']).__class__
                d_opt_config.pop('identifier')
                print(d_opt_config)
                d_optimizer = D_opt_class(**d_opt_config)

                adv_weight = config['discriminator']['weight']
                discr_args = dict(model=d_model, optimizer=d_optimizer, adv_weight=adv_weight, label_smoothing=config['discriminator']['label_smoothing'])

            
            

        ################################################################################################################
        ############################################# Model optimization ###############################################
        ################################################################################################################
            
            assert int(config['weight_pruning']) + int(config['weight_clustering']) + int(config['QAT']) <= 1, 'At most one weight modification can be applied at a time, to do collaborative optimization, you must train the model one after the other : ex :train prune -> train cluster -> train QAT'

            if config['weight_pruning']:
                prune_camera_inversion_layer = tfmot.sparsity.keras.prune_low_magnitude(gen_model.camera_inversion_layer)
                prune_perceptual_model = tfmot.sparsity.keras.prune_low_magnitude(gen_model.perceptual_model)

                gen_model.rebuild(camera_inversion_layer=prune_camera_inversion_layer,
                                    perceptual_model=prune_perceptual_model)
                
                gen_model = UnetModel(input_shape=input_shape,
                                    output_shape=output_shape,
                                    perceptual_args=prune_camera_inversion_layer,
                                    camera_inversion_args=prune_perceptual_model,
                                    name=config.model_name)
            

            if config['weight_clustering']:
                cluster_camera_inversion_layer = tfmot.clustering.keras.cluster_weights(gen_model.camera_inversion_layer,
                                                                                        **config['clustering_params'])
                cluster_perceptual_model = tfmot.clustering.keras.cluster_weights(gen_model.perceptual_model,
                                                                                        **config['clustering_params'])
                
                gen_model = UnetModel(input_shape=input_shape,
                                    output_shape=output_shape,
                                    perceptual_args=cluster_camera_inversion_layer,
                                    camera_inversion_args=cluster_perceptual_model,
                                    name=config.model_name)



            if config['load_pretrained']:
                print('loading pretrained model ...')
                gen_model.load_weights(config['pretrained_path_pb']).expect_partial()


            if config['QAT']:
                scope = {}
                qat_camera_inversion_layer = None
                if gen_model.camera_inversion_layer:
                    qat_camera_inversion_layer = quantize_annotate_layer(gen_model.camera_inversion_layer, InversionLayerQuantizeConfig())
                    qat_camera_inversion_layer = tf.keras.Sequential([Input(shape=input_shape, name='input', dtype='float32'),
                                                qat_camera_inversion_layer])
                    scope = {'InversionLayerQuantizeConfig': InversionLayerQuantizeConfig,
                                     'FTLayer': FTLayer}

                qat_perceptual_model = quantize_annotate_model(gen_model.perceptual_model)
                
                perceptual_model = qat_perceptual_model

                with quantize_scope(scope):
                    scheme = dict(QAT=tfmot.quantization.keras.default_8bit.Default8BitQuantizeScheme(),
                                  CQAT=tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(),
                                  PQAT=tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme(),
                                  PCQAT=tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(preserve_sparsity=True))
                    
                    qat_perceptual_model = quantize_apply(perceptual_model, scheme=scheme[config['QAT_scheme']])
                    if qat_camera_inversion_layer:
                        qat_camera_inversion_layer = quantize_apply(qat_camera_inversion_layer, scheme=scheme[config['QAT_scheme']])


                    gen_model = UnetModel(input_shape=input_shape,
                                    output_shape=output_shape,
                                    perceptual_args=qat_perceptual_model,
                                    camera_inversion_args=qat_camera_inversion_layer,
                                    name=config.model_name)  
                    
                    print(gen_model.summary())


        ################################################################################################################
        ############################################# Model compilation + callbacks + train ############################
        ################################################################################################################

            model = compile_model(gen_model=gen_model, 
                                gen_optimizer=optimizer, 
                                loss_dict=loss_dict, 
                                metrics=metrics, 
                                metric_weights=metric_weights, 
                                discr_args=discr_args,
                                in_shape=input_shape, 
                                out_shape=output_shape,
                                global_batch_size=config['batch_size'],#config['batch_size'],
                                distributed_gpu=config['distributed_gpu'],
                                num_gpus=len(gpus))
            
            print('='*70)
            print('='*70)
            print('='*70)
            print(model.summary())

            
            # to load the optimizer value
            # if config['load_pretrained']:
            #     print('loading pretrained model ...')
            #     model.load_weights(config['pretrained_path_pb']).expect_partial()

            #     if config['load_pretrained_optimizer']:
            #         test_model = tf.keras.models.load_model(config['pretrained_path_pb'], safe_mode=False,
            #                                         compile=False)
                    
            #         symbolic_weights = getattr(test_model.optimizer, 'variables')
            #         weight_values = K.batch_get_value(symbolic_weights)

            #         optimizer.build(model.trainable_variables)
            #         optimizer.set_weights(weight_values)
            #         model.optimizer = optimizer
                
                    # print(model.optimizer.variables)
                


            checkpoint_path = os.path.join(store_folder, 'checkpoints')
            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_path += '/'

            callbacks = get_callbacks(model=model, 
                                    store_folder=store_folder,
                                    checkpoint_path=checkpoint_path, 
                                    config=config, 
                                    dynamic_weights=dynamic_weights)
            


            model.fit(train,
                    epochs=config['epochs'],
                    callbacks=callbacks,
                    validation_data=val,
                    use_multiprocessing=True,
                    workers=config['workers'],
                    shuffle=True,
                    verbose=config['verbose'])

            model.load_weights(checkpoint_path).expect_partial()


            ################################################################################################################
            ############################################# Model saving ####################################################
            ################################################################################################################

            if config['save']:
                print('saving model ...')
                store_path = os.path.join(store_folder, 'models')
                if not os.path.isdir(store_path):
                    os.makedirs(store_path)
                


                name_gen = 'gen_' + config['model_name']
                if config['weight_pruning']:
                    name_gen += '_pruned'

                    gen_model = tfmot.sparsity.keras.strip_pruning(gen_model)


                    tf.keras.models.save_model(gen_model, '/home/jreymond/lensless_ml/temporary/gen_pruned.pb', include_optimizer=False)

                    tf.keras.models.save_model(gen_model.layers[1], '/home/jreymond/lensless_ml/temporary/pruned_perceptual.pb', include_optimizer=False)


                    perceptual = gen_model.layers[1]
                    converter = tf.lite.TFLiteConverter.from_keras_model(perceptual)
                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/pruned.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)

                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/pruned_weight.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)

                    
                if config['weight_clustering']:
                    name_gen += '_clustered'
                    # print('size before clustering: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')
                    gen_model = tfmot.clustering.keras.strip_clustering(gen_model)

                    tf.keras.models.save_model(gen_model, '/home/jreymond/lensless_ml/temporary/gen_cluster.pb', include_optimizer=False)

                    tf.keras.models.save_model(gen_model.layers[1], '/home/jreymond/lensless_ml/temporary/perceptual.pb', include_optimizer=False)




                name_gen += '.pb'

                tf.keras.models.save_model(model, os.path.join(store_path, "overal_model.pb"), include_optimizer=True)


                tf.keras.models.save_model(gen_model, os.path.join(store_path, name_gen), include_optimizer=True)

                if gen_model.camera_inversion_layer:
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




