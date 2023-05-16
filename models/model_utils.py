from models.unet import *
from models.flatnet.camera_inversion import *
from models.flatnet.gan import *
from models.flatnet.discriminator import *
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from custom_callbacks import *
from keras import Model

from keras_unet_collection import models as M_unet

from utils import *


class ReconstructionModel(Model):
    def __init__(self, output_shape, unet_args, camera_inversion_args=None, model_name='reconstruction model'):
        super().__init__(name=model_name)
        self.unet_args = unet_args
        self.camera_inversion_args = camera_inversion_args
        self.output_shape = output_shape


    def call(self, inputs):
        if self.camera_inversion_args:
            inputs = get_camera_inversion_layer(inputs, self.camera_inversion_args, mask=None)

        return u_net(inputs, **self.unet_args, out_shape=self.output_shape)

    


def get_model(config, input_shape, out_shape, model_name='Reconstruction model'):
    model_config = dict(config['model'][config['model_name']])
    input = Input(shape=input_shape, name='input', dtype='float32')

    x = input

    if config['use_camera_inversion']:
        x = get_camera_inversion_layer(input=x, config=config, mask=None)

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