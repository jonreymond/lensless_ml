from models.unet import *
from models.flatnet.camera_inversion import *
from models.flatnet.gan import *
from models.flatnet.discriminator import *
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from custom_callbacks import *

from utils import *





    

def get_model(config, input_shape, out_shape, model_name='Reconstruction model'):
    model_config = dict(config['model'][config['model_name']])
    # dummy_input = tf.zeros((config['batch_size'], *input_shape), dtype=tf.dtypes.float32)
    input = Input(shape=input_shape, name='input', dtype='float32')
    # input = Input(tensor=tf.convert_to_tensor(dummy_input), name='input', dtype='float32')
    x = input

    if config['use_camera_inversion']:
        x = get_camera_inversion_layer(input=x, config=config, mask=None)

    if model_config['type'] == 'unet':
        model_config.pop('type')
        x = u_net(input=x, **model_config, out_shape=out_shape)
        
    else:
        raise ValueError(f'Unknown model type: {model_config["type"]}')
    
    return Model(inputs=[input], outputs=[x], name=model_name)


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