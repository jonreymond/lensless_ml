from models.unet import *
from models.camera_inversion import *
from models.gan import *
from models.discriminator import *
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from custom_callbacks import *
from keras import Model
import keras

from keras.utils.layer_utils import count_params

from keras_unet_collection import models as M_unet

from utils import *

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.quantization.keras.quantizers import MovingAverageQuantizer, LastValueQuantizer
from tensorflow_model_optimization.quantization.keras import QuantizeConfig




class DistributedLossCombiner(Loss):
    def __init__(self, losses, loss_weights=None, name='total', global_batch_size=None, **kwargs):
        self.losses = losses
        self.loss_weights = loss_weights
        self.global_batch_size = global_batch_size
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)

    def call(self, y_true, y_pred):
        loss = tf.add_n([weight * tf.nn.compute_average_loss(loss(y_true, y_pred), global_batch_size=self.global_batch_size) for weight, loss in zip(self.loss_weights, self.losses)])

        # if model_losses:
        #     loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss


class DistributedLoss(Loss):
    def __init__(self, loss, name, global_batch_size, **kwargs):
        self.loss = loss
        self.global_batch_size = global_batch_size
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE, **kwargs)
        
    
    def call(self, y_true, y_pred):
        per_sample_loss = self.loss(y_true, y_pred)
        loss = tf.nn.compute_average_loss(per_sample_loss, global_batch_size=self.global_batch_size)
        # if model_losses:
        #     loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss
    




class ReconstructionModel2(Model):
    def __init__(self, input_shape, perceptual_model, camera_inversion_layer=None, name='reconstruction_model'):
        super().__init__(name=name)
        # self.unet_args = unet_args
        self.in_shape = input_shape
        self.camera_inversion_layer = camera_inversion_layer
        
        self.perceptual_model = perceptual_model


    def call(self, inputs):
        if self.camera_inversion_layer:
            inputs = self.camera_inversion_layer(inputs)

        return self.perceptual_model(inputs)
    
    def summary(self, **kwargs):
        cam_model = None
        if self.camera_inversion_layer:
            inp = Input(shape=self.in_shape, name='input', dtype='float32')

            print(inp.shape)
            out = self.camera_inversion_layer(inp)
            cam_model = Model(inputs=[inp], outputs=[out])
            cam_model.summary(**kwargs)
        self.perceptual_model.summary(**kwargs)

        if cam_model:
            model = keras.Sequential([cam_model, self.perceptual_model])
            line_length = 98
            print("=" * line_length)
            trainable_count = count_params(model.trainable_weights)
            non_trainable_count = count_params(model.non_trainable_weights)

            print(f"Total params: {trainable_count + non_trainable_count:,}")
            print(f"Trainable params: {trainable_count:,}")
            print(f"Non-trainable params: {non_trainable_count:,}")
            print("_" * line_length)



    


def get_model(config, input_shape, out_shape, model_name='Reconstruction model'):
    model_config = dict(config['model'][config['model_name']])
    input = Input(shape=input_shape, name='input', dtype='float32')

    x = input

    if config['use_camera_inversion']:
        x = get_camera_inversion_layer(config=config, mask=None)(x)

    if model_config['type'] == 'unet':
        model_config.pop('type')
        x = u_net(input=x, **model_config, out_shape=out_shape)
        
    elif model_config['type'] == 'unet_plus':
        model_config.pop('type')
        x = experimental_models(input=x, 
                      model_args=model_config['args'], 
                      out_shape=out_shape, 
                      model_name=config['model_name'])
    else:
        raise ValueError(f'Unknown model type: {model_config["type"]}')
    
    return Model(inputs=[input], outputs=[x], name=model_name)




MODELS = dict(unet_2d = M_unet.unet_2d,
            vnet_2d = M_unet.vnet_2d,
            unet_plus_2d = M_unet.unet_plus_2d,
            r2_unet_2d = M_unet.r2_unet_2d,
            att_unet_2d = M_unet.att_unet_2d,
            resunet_a_2d = M_unet.resunet_a_2d,
            u2net_2d = M_unet.u2net_2d,
            unet_3plus_2d = M_unet.unet_3plus_2d,
            transunet_2d = M_unet.transunet_2d,
            swin_unet_2d = M_unet.swin_unet_2d,
            unet = u_net)

# TODO : check if not 2 **unet_depth
def resize_input(dim, unet_depth):
    while dim % unet_depth != 0:
        dim += 1
    return dim


def experimental_models(input, model_args, out_shape, model_name):
    model_args = dict(model_args)

    _, height, width, channels = input.shape


    unet_depth = len(model_args['filter_num'])
    new_height = resize_input(height, unet_depth)
    new_width = resize_input(width, unet_depth)


    x = tf.keras.layers.Resizing(new_height, new_width)(x)
    
    model_args['n_labels'] = channels
    model_args['input_size'] = (new_height, new_width, channels)
    model_args['name'] = model_name
    
    gen_model = MODELS[model_name](**model_args)
    gen_model.summary()

    x = gen_model(x)
    x = tf.keras.layers.Resizing(height=out_shape[1], width=out_shape[2])(x)

    return x




def get_camera_inversion_params(gen_model):
    params = dict()
    # for variable in gen_model.trainable_variables:
    #     if 'camera_inversion' in variable.name:
    #         params[variable.name] = variable.numpy()
    # return params

    for layer in gen_model.layers:
        if layer.name.startswith('separable_layer'):
            params['activation'] = layer.activation
            params['W1'] = layer.W1.numpy()
            params['W2'] = layer.W2.numpy()
            break
        elif layer.name.startswith('non_separable_layer'):
            # ft_layer, normalizer, crops, pad_input=None, pad_psf=None, mask=None
            params['ft_layer'] = layer.ft_layer.numpy()
            params['normalizer'] = layer.normalizer.numpy()

            params['low_crop_h'] = layer.crops['low_crop_h']
            params['high_crop_h'] = layer.crops['high_crop_h']
            params['low_crop_w'] = layer.crops['low_crop_w']
            params['high_crop_w'] = layer.crops['high_crop_w']

            params['pad_input'] = layer.pad_input
            params['pad_psf'] = layer.pad_psf
            params['mask'] = layer.mask.numpy()
            break
    return params


# def get_camera_inversion_layer(gen_model):
#     for layer in gen_model.layers:
#         if layer.name.startswith('separable_layer') or layer.name.startswith('non_separable_layer'):
#             return layer
#     return None

def get_perceptual_model(gen_model, excluded_layers=['input', 'non_separable_layer', 'separable_layer'], verbose=True):
    begin_perceptual_idx = None

    for idx, layer in enumerate(gen_model.layers):
        if not any([layer.name.startswith(excluded_layer) for excluded_layer in excluded_layers]):
            if verbose: print('Begin new model:', layer.name, ', input shape:', layer.input_shape)
            begin_perceptual_idx = idx

            break
        else:
            if verbose: print('- layer not taken:', layer.name)
        
    perceptual_model = tf.keras.models.Model(inputs=gen_model.layers[begin_perceptual_idx].input, outputs=gen_model.layers[-1].output)
    if verbose: perceptual_model.summary()
    return perceptual_model




def get_metrics(config, out_shape, batch_size):
    """get the metrics from the config

    Args:
        config (dict): configuration dictionary

    Returns:
        (list, list): the metrics and their corresponding weights
    """
    metrics = []
    metric_weights = []

    for metric_id in config['metric'].keys():
        metric_config = config['metric'][metric_id]
        metric_args = dict(shape=out_shape, batch_size=batch_size, model=metric_config['model']) if metric_id == 'lpips' else None
        metrics.append(get_loss_from_name(metric_id, config['distributed_gpu'], metric_args))
        metric_weights.append(metric_config['weight'])

    return metrics, metric_weights


def get_losses(config, out_shape, batch_size):
    """get the losses from the config

    Args:
        config (dict): configuration dictionary

    Returns:
        (list, list, list): the losses and their corresponding weights, and the dynamic weights
                            that need to be updated during training
    """
    loss_dict = dict()

    dynamic_weights = []
    for loss_id in config['loss'].keys():
        loss_config = config['loss'][loss_id]

        weight = loss_config['weight']
        # add to list to be updated during callback
        if loss_config['additive_factor']:
            weight = K.variable(weight)
            dynamic_weights.append((weight, loss_config['additive_factor']))

        loss_args = dict(shape=out_shape, batch_size=batch_size, model=loss_config['model']) if loss_id == 'lpips' else None
        loss_dict.update({loss_id: (weight, get_loss_from_name(loss_id, config['distributed_gpu'], loss_args))})

    return loss_dict, dynamic_weights


def compile_model(gen_model, gen_optimizer, loss_dict, metrics, metric_weights, discr_args=None, in_shape=None, out_shape=None, global_batch_size=None, distributed_gpu=False, num_gpus=1):
    loss_weights, losses = zip(*list(loss_dict.values()))


    if not distributed_gpu:
        total_loss = LossCombiner(losses, loss_weights, name='total')
    else :
        global_batch_size = global_batch_size if discr_args else global_batch_size // num_gpus
        total_loss = DistributedLossCombiner(losses=losses, 
                                             loss_weights=loss_weights, 
                                             name='total', 
                                             global_batch_size=global_batch_size)
        new_metrics = []
        for m in metrics:
            new_metrics.append(DistributedLoss(m, m.name, global_batch_size=global_batch_size))
        metrics = new_metrics            
    
    if not discr_args:

        gen_model.compile(optimizer=gen_optimizer, 
                             loss=total_loss, 
                             metrics=[*metrics, total_loss])
        model = gen_model
    
    else:
        model = FlatNetGAN(discriminator=discr_args['model'], generator=gen_model, global_batch_size=global_batch_size, label_smoothing=discr_args['label_smoothing'])        

        model.compile(optimizer=gen_optimizer,
                    d_optimizer=discr_args['optimizer'],
                    adv_weight=discr_args['adv_weight'],
                    mse_weight=loss_dict['mse'][0],
                    lpips_loss=loss_dict['lpips'][1],
                    mse_loss = loss_dict['mse'][1],
                    perc_weight=loss_dict['lpips'][0],
                    metrics=[*metrics, total_loss])
        model.build(Input(shape=in_shape).shape)

    return model


def get_callbacks(model, store_folder, checkpoint_path, dynamic_weights, config):

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=checkpoint_path,
                                            save_weights_only=True,
                                            monitor='val_total',
                                            mode='min',
                                            save_best_only=True,
                                            save_freq="epoch",
                                            verbose=1)
    
    callbacks = [model_checkpoint]

    if config['weight_pruning']:
        callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())
        callbacks.append(tfmot.sparsity.keras.PruningSummaries(log_dir=os.path.join(store_folder, 'pruning_logs')))

    if dynamic_weights:
        print('Using dynamic weights')
        callbacks.append(ChangeLossWeights(dynamic_weights))

    if config['lr_reducer']['type']:
        reducer_args = config['lr_reducer']
        if reducer_args['type'] == "reduce_lr_on_plateau":
            callbacks.append(ReduceLROnPlateau(**reducer_args['reduce_lr_on_plateau']))
        elif reducer_args['type'] == "learning_rate_scheduler":
            callbacks.append(LearningRateScheduler(**reducer_args['learning_rate_scheduler']))
        else:
            raise ValueError(reducer_args['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")
        
    if config['use_discriminator']:
        assert not (config['discriminator']['optimizer']['use_lr_reducer'] and config['discriminator']['optimizer']['copy_gen_lr']), 'Cannot use both lr reducer and copy gen lr' 
        if config['discriminator']['optimizer']['use_lr_reducer']:
            print('Using discriminator lr reducer')
            reducer_args = config['discriminator']['optimizer']['lr_reducer']
            if reducer_args['type'] == "reduce_lr_on_plateau":
                callbacks.append(ReduceLROnPlateauCustom(model.d_optimizer, **reducer_args['reduce_lr_on_plateau']))
            elif reducer_args['type'] == "learning_rate_scheduler":
                callbacks.append(LearningRateSchedulerCustom(model.d_optimizer, **reducer_args['learning_rate_scheduler']))
            else:
                raise ValueError(reducer_args['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")
        elif config['discriminator']['optimizer']['copy_gen_lr']:
            print('Using discriminator lr copy')
            callbacks.append(CopyLearningRate(model.d_optimizer, model.optimizer))
            


    if config['use_tensorboard']:
        tb_path = os.path.join(store_folder, 'tensorboard_logs')
        if not os.path.isdir(tb_path):
            os.makedirs(tb_path)
        tb_callback = tf.keras.callbacks.TensorBoard(tb_path, 
                                                     histogram_freq = 1, 
                                                     profile_batch=config['tensorboard_profile_batch'],
                                                     update_freq='epoch')
        callbacks.append(tb_callback)
    
    return callbacks






class InversionLayerQuantizeConfig(QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):

        weights_and_quantizers = [(layer_weights, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))
                                  for layer_weights in layer.get_list_weights()]
        
        return weights_and_quantizers
                
    

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.set_list_weights(quantize_weights)

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]


    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []

    def get_config(self):
      return {}
    




class ReconstructionModel3(keras.Sequential):
    def __init__(self, input_shape, perceptual_model, camera_inversion_layer=None, name='reconstruction_model'):
        super().__init__(name=name)
        # self.unet_args = unet_args
        self.in_shape = input_shape

        self.camera_inversion_layer = camera_inversion_layer
        self.perceptual_model = perceptual_model
        
        layers = [Input(shape=input_shape, name='input', dtype='float32'), self.perceptual_model]
        if self.camera_inversion_layer:
            if isinstance(self.camera_inversion_layer, tf.keras.Model):
                self.camera_inversion_layer.build(input_shape=self.in_shape)
            else:
                layers.insert(1, self.camera_inversion_layer)

        super().__init__(name=name, layers=layers)


    
    def summary(self, **kwargs):
        cam_model = None
        if self.camera_inversion_layer:
            inp = Input(shape=self.in_shape, name='input', dtype='float32')
            out = self.camera_inversion_layer(inp)
            cam_model = Model(inputs=[inp], outputs=[out])
            if isinstance(self.camera_inversion_layer, tf.keras.Model):
                self.camera_inversion_layer.summary(**kwargs)
            else:
                cam_model.summary(**kwargs)

        self.perceptual_model.summary(**kwargs)

        if cam_model:
            model = keras.Sequential([cam_model, self.perceptual_model])
            line_length = 98
            print("=" * line_length)
            trainable_count = count_params(model.trainable_weights)
            non_trainable_count = count_params(model.non_trainable_weights)

            print(f"Total params: {trainable_count + non_trainable_count:,}")
            print(f"Trainable params: {trainable_count:,}")
            print(f"Non-trainable params: {non_trainable_count:,}")
            print("_" * line_length)