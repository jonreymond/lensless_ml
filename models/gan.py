from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.losses import MeanSquaredError
from keras.losses import Loss
from keras import optimizers

import models.model_utils as model_utils


###############################################################################
############################## GAN Model ######################################
###############################################################################

class DiscrLoss(Loss):
    def __init__(self, name='discr_loss', label_smoothing=0, **kwargs):
        super().__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
       

    def call(self, y_true, y_pred):
        return -tf.math.log(self.label_smoothing + y_true) - tf.math.log((1 - self.label_smoothing) - y_pred)
        # return tf.math.reduce_mean(-tf.math.log(y_true) - tf.math.log(1 - y_pred))



class FlatNetGAN(Model):
    def __init__(self, discriminator, generator, global_batch_size=None, label_smoothing=0, **kwargs):
        super(FlatNetGAN, self).__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.global_batch_size = global_batch_size
        self.label_smoothing = label_smoothing
        # losses
        self.d_loss = model_utils.DistributedLoss(DiscrLoss(label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.NONE),
                                                  name='discr', 
                                                  global_batch_size=global_batch_size)
        

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

        # Discriminator training
        gen_img = self.generator(sensor_img)
        with tf.GradientTape() as tape:
            pred_gen = self.discriminator(gen_img)
            # tf.print('pred gen', pred_gen)
            pred_real = self.discriminator(real_img)
            # tf.print('pred real', pred_real)
            d_loss_res = self.d_loss(pred_real, pred_gen)

        grads = tape.gradient(d_loss_res, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        # Generator training
        with tf.GradientTape() as tape:
            gen_img = self.generator(sensor_img)
            adv_loss = tf.nn.compute_average_loss(-tf.math.log(self.discriminator(gen_img)), global_batch_size=self.global_batch_size)
            mse_loss = self.g_mse_loss(real_img, gen_img)
            perc_loss = self.lpips_loss(real_img, gen_img)
            
            g_loss_res = self.adv_weight * adv_loss + self.mse_weight * mse_loss + self.perc_weight * perc_loss

        grads = tape.gradient(g_loss_res, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # tf.print('d_loss', d_loss_res)
        # tf.print('g_loss', g_loss_res)
        # tf.print('adv_loss', adv_loss)
        # tf.print('mse_loss', mse_loss)
        # tf.print('perc_loss', perc_loss)
        
        return {"d": d_loss_res, "g": g_loss_res, "adv": adv_loss, "mse":mse_loss, "lpips" : perc_loss}
    
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







