from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.losses import MeanSquaredError
from keras.losses import Loss
from keras import optimizers

import models.model_utils as model_utils
import sys


###############################################################################
############################## GAN Model ######################################
###############################################################################

class DiscrLoss(Loss):
    def __init__(self, name='discr_loss', label_smoothing=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, **kwargs)


    def call(self, y_true, y_pred):
        # put into range [0, 1] --> now both in [-1, 1]
        # y_true = (y_true + 1) /2
        real_loss = self.cross_entropy(tf.ones_like(y_true) - self.label_smoothing, y_true)
        fake_loss = self.cross_entropy(tf.zeros_like(y_pred) + self.label_smoothing, y_pred)
        total_loss = real_loss + fake_loss
        return total_loss
    
        # return -tf.math.log(self.label_smoothing + y_true) - tf.math.log((1 - self.label_smoothing) - y_pred)
        # return tf.math.reduce_mean(-tf.math.log(y_true) - tf.math.log(1 - y_pred))

class AdversarialLoss(Loss):
    def __init__(self, name='adv_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, **kwargs)

    def call(self, y_true, y_pred):
        loss = self.cross_entropy(tf.ones_like(y_pred), y_pred)

        return loss



class FlatNetGAN(Model):
    def __init__(self, discriminator, generator, global_batch_size=None, label_smoothing=0, **kwargs):
        super(FlatNetGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.global_batch_size = global_batch_size
        self.label_smoothing = label_smoothing
        # losses
        # self.d_loss = model_utils.DistributedLoss(DiscrLoss(label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.NONE),
        #                                           name='discr', 
        #                                           global_batch_size=global_batch_size)
        
        # self.g_adv_loss = model_utils.DistributedLoss(AdversarialLoss(reduction=tf.keras.losses.Reduction.NONE),
        #                                             name='adv',
        #                                             global_batch_size=global_batch_size)
        # TODO : don't work with multi-gpu (check why return scalar even with reduction=None)
        self.g_adv_loss = AdversarialLoss(name='adv')
        self.d_loss = DiscrLoss(name='discr', label_smoothing=label_smoothing)
        

    def compile(self, optimizer, d_optimizer, lpips_loss, mse_loss, adv_weight, mse_weight, perc_weight, metrics):
        super(FlatNetGAN, self).compile(metrics=metrics, optimizer=optimizer)
        self.d_optimizer = optimizers.get(d_optimizer) if isinstance(d_optimizer, str) else d_optimizer
        self.g_optimizer = optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer

        self.lpips_loss = model_utils.DistributedLoss(lpips_loss, lpips_loss.name, self.global_batch_size)
        self.g_mse_loss = model_utils.DistributedLoss(mse_loss, mse_loss.name, self.global_batch_size)
        self.adv_weight = adv_weight
        self.mse_weight = mse_weight
        self.perc_weight = perc_weight
        

    # TODO : check
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







