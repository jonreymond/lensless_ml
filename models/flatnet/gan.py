from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.losses import MeanSquaredError
from keras.losses import Loss
from keras import optimizers



###############################################################################
############################## GAN Model ######################################
###############################################################################

class DiscrLoss(Loss):
    def __init__(self, name='discr_loss'):
        super().__init__(name=name)
       

    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(-tf.math.log(y_true) - tf.math.log(1 - y_pred))


class FlatNetGAN(Model):
    def __init__(self, discriminator, generator):
        super(FlatNetGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        # losses
        self.d_loss = DiscrLoss()


    def compile(self, optimizer, d_optimizer, lpips_loss, adv_weight, mse_weight, perc_weight, metrics):
        super(FlatNetGAN, self).compile(metrics=metrics, optimizer=optimizer)
        self.d_optimizer = optimizers.get(d_optimizer) if isinstance(d_optimizer, str) else d_optimizer
        self.g_optimizer = optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer

        self.lpips_loss = lpips_loss
        self.g_mse_loss = MeanSquaredError()
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
            adv_loss = tf.math.reduce_mean(- tf.math.log(self.discriminator(gen_img)))
            mse_loss = self.g_mse_loss(real_img, gen_img)
            perc_loss = self.lpips_loss(real_img, gen_img)
            g_loss_res = self.adv_weight * adv_loss + self.mse_weight * mse_loss + self.perc_weight * perc_loss

        grads = tape.gradient(g_loss_res, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"d": d_loss_res, "g": g_loss_res, "adv": adv_loss, "mse":mse_loss, "lpips" : perc_loss}
    







