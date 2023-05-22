from models.unet import *
from models.flatnet.camera_inversion import *
from models.flatnet.gan import *
from models.flatnet.discriminator import *
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from custom_callbacks import *
from keras import Model
import keras

from keras.utils.layer_utils import count_params

from keras_unet_collection import models as M_unet

from utils import *


class ReconstructionModel(Model):
    def __init__(self, input_shape, output_shape, unet_args, camera_inversion_args=None, model_name='reconstruction_model'):
        super().__init__(name=model_name)
        # self.unet_args = unet_args
        self.in_shape = input_shape
        self.camera_inversion_layer = get_camera_inversion_layer(camera_inversion_args) if camera_inversion_args else None

        input = Input(shape=input_shape, name='input', dtype='float32')
        if self.camera_inversion_layer:
            input = self.camera_inversion_layer(input)
        
        self.perceptual_model = Model(inputs=[input], 
                                      outputs=[u_net(input=input, **unet_args, out_shape=output_shape)],
                                      name='perceptual_model')


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




EXPERIMENTAL_MODELS = dict(unet_2d = M_unet.unet_2d,
                           vnet_2d = M_unet.vnet_2d,
                           unet_plus_2d = M_unet.unet_plus_2d,
                           r2_unet_2d = M_unet.r2_unet_2d,
                           att_unet_2d = M_unet.att_unet_2d,
                           resunet_a_2d = M_unet.resunet_a_2d,
                           u2net_2d = M_unet.u2net_2d,
                           unet_3plus_2d = M_unet.unet_3plus_2d,
                           transunet_2d = M_unet.transunet_2d,
                           swin_unet_2d = M_unet.swin_unet_2d)

def resize_input(dim, unet_depth):
    while dim % unet_depth != 0:
        dim += 1
    return dim

def experimental_models(input, model_args, out_shape, model_name):
    model_args = dict(model_args)

    _, num_channels, height, width = input.shape
    print('num_channels', num_channels)
    print('height', height)
    print('width', width)
    x = to_channel_last(input)
    unet_depth = len(model_args['filter_num'])
    new_height = resize_input(height, unet_depth)
    new_width = resize_input(width, unet_depth)
    print('new_height', new_height)
    print('new_width', new_width)

    x = tf.keras.layers.Resizing(new_height, new_width)(x)
    
    model_args['n_labels'] = num_channels
    model_args['input_size'] = (new_height, new_width, num_channels)
    model_args['name'] = model_name
    
    gen_model = EXPERIMENTAL_MODELS[model_name](**model_args)
    gen_model.summary()

    x = gen_model(x)
    print(out_shape)
    x = tf.keras.layers.Resizing(height=out_shape[1], width=out_shape[2])(x)
    output = to_channel_first(x)
    return output




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




def get_metrics(config):
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
        metrics.append(get_loss_from_name(metric_id, metric_config, config))
        metric_weights.append(metric_config['weight'])

    return metrics, metric_weights


def get_losses(config):
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

        loss_dict.update({loss_id: (weight, get_loss_from_name(loss_id, loss_config, config))})

    return loss_dict, dynamic_weights


def compile_model(gen_model, gen_optimizer, loss_dict, metrics, metric_weights, discr_args=None, in_shape=None, out_shape=None):
    if not discr_args:
        loss_weights, losses = zip(*list(loss_dict.values()))
        loss = LossCombiner(losses, loss_weights, name='loss_comb')
        gen_model.compile(optimizer=gen_optimizer, 
                             loss=loss, 
                             metrics=[*metrics, LossCombiner(metrics, metric_weights, name='total')])
        model = gen_model
    
    else:
        model = FlatNetGAN(discriminator=discr_args['model'], generator=gen_model)

        lpips_loss = loss_dict['lpips'][1]

        model.compile(optimizer=gen_optimizer,
                    d_optimizer=discr_args['optimizer'],
                    adv_weight=discr_args['adv_weight'],
                    mse_weight=loss_dict['mse'][0],
                    lpips_loss=lpips_loss,
                    perc_weight=loss_dict['lpips'][0],
                    metrics=[MeanSquaredError(name='mse'), 
                                lpips_loss, 
                                LossCombiner([lpips_loss, MeanSquaredError(name='mse')], [1.2, 1], name='total')])
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
    if dynamic_weights:
        callbacks.append(ChangeLossWeights(dynamic_weights))

    if config['lr_reducer']['type']:
        reducer_args = config['lr_reducer']
        if reducer_args['type'] == "reduce_lr_on_plateau":
            callbacks.append(ReduceLROnPlateau(**reducer_args['reduce_lr_on_plateau']))
        elif reducer_args['type'] == "learning_rate_scheduler":
            callbacks.append(LearningRateScheduler(**reducer_args['learning_rate_scheduler']))
        else:
            raise ValueError(reducer_args['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")
        
    if config['use_discriminator'] and config['discriminator']['optimizer']['use_lr_reducer']:
        reducer_args = config['discriminator']['optimizer']['lr_reducer']
        if reducer_args['type'] == "reduce_lr_on_plateau":
            callbacks.append(ReduceLROnPlateauCustom(model.d_optimizer, **reducer_args['reduce_lr_on_plateau']))
        elif reducer_args['type'] == "learning_rate_scheduler":
            callbacks.append(LearningRateSchedulerCustom(model.d_optimizer, **reducer_args['learning_rate_scheduler']))
        else:
            raise ValueError(reducer_args['type'] + " type is not supported, must be either 'reduce_lr_on_plateau' or 'learning_rate_scheduler'")


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



from tensorflow_model_optimization.quantization.keras.quantizers import MovingAverageQuantizer, LastValueQuantizer
from tensorflow_model_optimization.quantization.keras import QuantizeConfig


class FTLayerQuantizeConfig(QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        weights_and_quantizers =  [(layer.ft_layer, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)),
                                   (layer.normalizer, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]
        if layer.mask:
            weights_and_quantizers.append((layer.mask, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)))
        return weights_and_quantizers
                
    

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.ft_layer = quantize_weights[0]
        layer.normalizer = quantize_weights[1]
        if layer.mask:
            layer.mask = quantize_weights[2]

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