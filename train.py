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

# from dataset import get_dataset
from models.unet import u_net
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from models.model_utils import *



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
        # To record terminal output:

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


            loss_dict, dynamic_weights = get_losses(config, output_shape, config['batch_size'])
            metrics, metric_weights = get_metrics(config, output_shape, config['batch_size'])


            opt_config = dict(config['optimizer'])
            Opt_class = tf.keras.optimizers.get(opt_config['identifier']).__class__
            opt_config.pop('identifier')
            optimizer = Opt_class(**opt_config)
      


            ################################################################################################################
            ###################################### Model generation ########################################################
            ################################################################################################################


            input = Input(shape=input_shape, name='input', dtype='float32')
            # camera inversion layer #
            camera_inversion_layer = None
            if config['use_camera_inversion']:
                # Separable dataset
                if dataset_config['type_mask'] == 'separable':
                    if config['random_init']:
                        raise NotImplementedError('Random init not implemented for separable dataset')
                    else:
                        # assert dataset_config['name'] == 'flatnet', 'Only flatnet dataset is supported for separable dataset'
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


        ################################################################################################################
        ############################################# Discriminator setup ##############################################
        ################################################################################################################

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

            
            

        ################################################################################################################
        ############################################# Model optimization ###############################################
        ################################################################################################################
            
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

            
            # gen_model = tf.keras.models.load_model(config['pretrained_path_pb'], safe_mode=False,
            #                                         compile=False)
            


            # # print(gen_model.summary())
            # gen_model = tfmot.sparsity.keras.strip_pruning(gen_model)
            # # print(gen_model.summary())
            # gen_model = ReconstructionModel3(input_shape=input_shape,
            #                                     camera_inversion_layer=gen_model.layers[0],
            #                                     perceptual_model=gen_model.layers[1])

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


                    gen_model = ReconstructionModel3(input_shape=input_shape,
                                                    camera_inversion_layer=qat_camera_inversion_layer,
                                                    perceptual_model=qat_perceptual_model)
                    print(gen_model.summary())



            print(get_model_memory_usage(batch_size=1,
                                         model=gen_model.perceptual_model))
            print(get_model_memory_usage(batch_size=config['batch_size'], model=tf.keras.Sequential([Input(shape=input_shape, name='input', dtype='float32'),
                                                gen_model.camera_inversion_layer])))


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

            print(print('size before pruning: ', get_gzipped_model_size(gen_model, 'mb'), 'MB'))


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
                    # print('size after pruning: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')


                    tf.keras.models.save_model(gen_model, '/home/jreymond/lensless_ml/temporary/gen_pruned.pb', include_optimizer=False)

                    tf.keras.models.save_model(gen_model.layers[1], '/home/jreymond/lensless_ml/temporary/pruned_perceptual.pb', include_optimizer=False)

                    
                    # perceptual_model=gen_model.layers[1]

                    perceptual = gen_model.layers[1]
                    converter = tf.lite.TFLiteConverter.from_keras_model(perceptual)
                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/pruned.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)

                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/pruned_weight.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)

                    print('size after pruning: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')
                    
                    

                    
                if config['weight_clustering']:
                    name_gen += '_clustered'
                    # print('size before clustering: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')
                    gen_model = tfmot.clustering.keras.strip_clustering(gen_model)
                    print('size after clustering: ', get_gzipped_model_size(gen_model, 'mb'), 'MB')

                    tf.keras.models.save_model(gen_model, '/home/jreymond/lensless_ml/temporary/gen_cluster.pb', include_optimizer=False)

                    tf.keras.models.save_model(gen_model.layers[1], '/home/jreymond/lensless_ml/temporary/perceptual.pb', include_optimizer=False)

                    # perceptual_model=gen_model.layers[1]

                    perceptual = gen_model.layers[1]
                    converter = tf.lite.TFLiteConverter.from_keras_model(perceptual)
                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/cluster.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)

                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/cluster_weight.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)

                    converter.target_spec.supported_ops = [ 
                                                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                                # tf.lite.OpsSet.TFLITE_BUILTINS,
                                                # tf.lite.OpsSet.SELECT_TF_OPS
                                                  ] 
                    converter.representative_dataset = lambda: (i for (i, j) in val.take(4))

                    tflite_clustered_model = converter.convert()
                    with open('/home/jreymond/lensless_ml/temporary/cluster_weight_quant.tflite', 'wb') as f:
                        f.write(tflite_clustered_model)



                name_gen += '.pb'

                tf.keras.models.save_model(model, os.path.join(store_path, "overal_model.pb"), include_optimizer=True)


                tf.keras.models.save_model(gen_model, os.path.join(store_path, name_gen), include_optimizer=True)

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




