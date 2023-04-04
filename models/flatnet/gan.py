from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.losses import MeanSquaredError



###############################################################################
############################## GAN Model ######################################
###############################################################################

class FlatNetGAN(Model):
    def __init__(self, discriminator, generator):
        super(FlatNetGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        # losses
        self.d_loss = lambda d_true, d_fake : tf.math.log(d_true) - tf.math.log(1 - d_fake)


    def compile(self, d_optimizer, g_optimizer, g_perceptual_loss, adv_weight, mse_weight, perc_weight):
        super(FlatNetGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.g_percept_loss = g_perceptual_loss
        self.g_mse_loss = MeanSquaredError()
        self.adv_weight = adv_weight
        self.mse_weight = mse_weight
        self.perc_weight = perc_weight


    def train_step(self, inputs):
        sensor_img, real_img = inputs

        # Discriminator training
        gen_img = self.generator(sensor_img)
        with tf.GradientTape() as tape:
            pred_gen = self.discriminator(gen_img)
            pred_real = self.discriminator(real_img)
            d_loss_res = self.loss_d(pred_real, pred_gen)

        grads = tape.gradient(d_loss_res, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))


        # Generator training
        with tf.GradientTape() as tape:
            gen_img = self.generator(sensor_img)
            adv_loss = - tf.math.log(self.discriminator(gen_img))
            mse_loss = self.g_mse_loss(real_img, gen_img)
            perc_loss = self.g_perceptual_loss(real_img, gen_img)
            g_loss_res = self.adv_weight * adv_loss + self.mse_weight * mse_loss + self.perc_weight * perc_loss

        grads = tape.gradient(g_loss_res, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        return {"d_loss": d_loss_res, "g_loss": g_loss_res}
    







