# #############################################################################
# gan_model.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################



import tensorflow as tf

import keras
from keras import Model
from keras.layers import Input, BatchNormalization, GroupNormalization, GlobalAveragePooling2D, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.losses import Loss
from keras import optimizers


from utils.utils import LossCombiner, DistributedLossCombiner, DistributedLoss




######################################### Discriminator #########################################################

class Discriminator(keras.Sequential):
    def __init__(self,
                 input_shape, 
                 filters, 
                 strides, 
                 kernel_size, 
                 activation='swish', 
                 use_groupnorm=False, 
                 num_groups=None, 
                 sigmoid_output=False,
                 name='discriminator'):
        
        assert activation, "activation must be specified"
        input = Input(shape=input_shape, name="input")
        x = input

        assert len(filters) == len(strides) and len(strides) == len(kernel_size)

        for i in range(len(filters)):
            # conv block
            x = Conv2D(filters[i],
                    kernel_size=kernel_size[i],
                    strides=strides[i],
                    padding='same',
                    )(x)
            if use_groupnorm:
                x = GroupNormalization(groups=num_groups)(x)
            else:
                x = BatchNormalization()(x)
            
            x = Activation(activation=activation)(x)
            

        x = GlobalAveragePooling2D(keepdims=True)(x)

        x = Conv2D(1, kernel_size=1, padding='same', activation=None)(x)
        x = Reshape(target_shape=[])(x)

        if sigmoid_output:
            x = Activation("sigmoid")(x)
        
        m = Model(inputs=[input], outputs=[x], name='discriminator')
        super().__init__(name=name, layers=m.layers)



######################################## Discriminator/GAN Loss ##################################################



class DiscrLoss(Loss):
    def __init__(self, name='discr_loss', label_smoothing=None, **kwargs):
        """Discriminator loss to train the discriminator model

        Args:
            name (str, optional): name of the loss. Defaults to 'discr_loss'.
            label_smoothing (dict, optional): label smoothing parameters. Defaults to None.
        """
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, **kwargs)


    def call(self, y_true, y_pred):
        zeros = tf.zeros_like(y_pred)
        ones = tf.ones_like(y_pred)
        if self.label_smoothing:
            zeros = tf.random.uniform(shape=tf.shape(y_pred), 
                                       minval=self.label_smoothing['fake_range'][0], 
                                       maxval=self.label_smoothing['fake_range'][1])
            ones = tf.random.uniform(shape=tf.shape(y_pred), 
                                       minval=self.label_smoothing['true_range'][0], 
                                       maxval=self.label_smoothing['true_range'][1])
            
        real_loss = self.cross_entropy(ones, y_true)
        fake_loss = self.cross_entropy(zeros, y_pred)
        total_loss = real_loss + fake_loss
        return total_loss




class AdversarialLoss(Loss):
    def __init__(self, name='adv_loss', **kwargs):
        """Adversarial loss to train the generator model, the input y_pred is the output of the discriminator model

    Args:
        name (str, optional): name of the loss. Defaults to 'adv_loss'.
    """
        super().__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, **kwargs)

    def call(self, y_true, y_pred):
        loss = self.cross_entropy(tf.ones_like(y_pred), y_pred)

        return loss




######################################### GAN ################################################################




class FlatNetGAN(Model):
    def __init__(self, discriminator, generator, global_batch_size=None, label_smoothing=None, **kwargs):
        """GAN model using a discriminator and a generator
        
        Args:
            discriminator (tf.keras.Model): discriminator model
            generator (tf.keras.Model): generator model
            global_batch_size (int, optional): global batch size. Defaults to None.
            label_smoothing (dict, optional): label smoothing parameters for the discriminator training. Defaults to None.
            """
        super(FlatNetGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.global_batch_size = global_batch_size
        self.label_smoothing = label_smoothing

        self.g_adv_loss = AdversarialLoss(name='adv')
        self.d_loss = DiscrLoss(name='discr', label_smoothing=label_smoothing)
        

    def compile(self, optimizer, d_optimizer, lpips_loss, mse_loss, adv_weight, mse_weight, perc_weight, metrics, distributed_gpu=False):
        super(FlatNetGAN, self).compile(metrics=metrics, optimizer=optimizer)
        self.d_optimizer = optimizers.get(d_optimizer) if isinstance(d_optimizer, str) else d_optimizer
        self.g_optimizer = optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer


        self.lpips_loss = lpips_loss
        self.g_mse_loss = mse_loss
        self.adv_weight = adv_weight
        self.mse_weight = mse_weight
        self.perc_weight = perc_weight
        
    def call(self, inputs):
        return self.generator(inputs)


    def train_step(self, inputs):
        sensor_img, real_img = inputs

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_img = self.generator(sensor_img, training=True)
            real_output = self.discriminator(real_img, training=True)
            fake_output = self.discriminator(gen_img, training=True)

            adv_loss = self.g_adv_loss(None, fake_output)
            mse_loss = self.g_mse_loss(real_img, gen_img)
            perc_loss = self.lpips_loss(real_img, gen_img)
            gen_loss = self.adv_weight * adv_loss + self.mse_weight * mse_loss + self.perc_weight * perc_loss

            disc_loss = self.d_loss(real_output, fake_output)
            

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return {"d": disc_loss, "g": gen_loss, "adv": adv_loss, "mse":mse_loss, "lpips" : perc_loss}
    
    def summary(self, **kwargs):

        self.generator.summary(**kwargs)
        self.discriminator.summary(**kwargs)
    
    def get_config(self):
        config = super().get_config()

        config.update({
            "discriminator": self.discriminator, 
            "generator": self.generator, 
            "global_batch_size": self.global_batch_size,
            "label_smoothing": self.label_smoothing
        })
        return config
    


######################################### GAN-Model compilation ####################################################



def compile_model(gen_model, gen_optimizer, loss_dict, metrics, metric_weights, discr_args=None, in_shape=None, out_shape=None, global_batch_size=None, distributed_gpu=False, num_gpus=1):
    loss_weights, losses = zip(*list(loss_dict.values()))


    if not distributed_gpu:
        total_loss = LossCombiner(losses, loss_weights, name='total')
        total_metric = LossCombiner(metrics, metric_weights, name='total')
    else :
        global_batch_size = global_batch_size if discr_args else global_batch_size // num_gpus
        total_loss = DistributedLossCombiner(losses=losses, 
                                             loss_weights=loss_weights, 
                                             name='total', 
                                             global_batch_size=global_batch_size)
        
        total_metric = DistributedLossCombiner(losses=metrics,
                                                loss_weights=metric_weights,
                                                name='total',
                                                global_batch_size=global_batch_size)
        new_metrics = []
        for m in metrics:
            new_metrics.append(DistributedLoss(m, m.name, global_batch_size=global_batch_size))
        metrics = new_metrics            
    
    if not discr_args:

        gen_model.compile(optimizer=gen_optimizer, 
                             loss=total_loss, 
                             metrics=[*metrics, total_metric])
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
                    metrics=[*metrics, total_loss],
                    distributed_gpu=distributed_gpu)
        model.build(Input(shape=in_shape).shape)

    return model



def lr_scheduler(epoch, lr, epochs_interval, factor, min_lr):

    if (epoch +1) % epochs_interval == 0 and epoch > 0:
        return max(lr * factor, min_lr)
    else:
        return lr