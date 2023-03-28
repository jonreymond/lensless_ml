from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation

from keras.models import Model
import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import MeanSquaredError



#####################################################################
###################### Discriminator ################################
#####################################################################

def conv_block(x, filters, kernel_size, strides=(1,1), activation='relu', num_groups=None):
    x = Conv2D(filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same', # TODO : check
                )(x)
    if num_groups:
        x = GroupNormalization(groups=num_groups)(x)
    
    x = Activation(activation=activation)(x)
    return x
    

def get_discriminator(shape, num_groups):
    input = Input(shape=shape, name="input")
    x = input

    filters = [64, 128, 128, 256]
    x = conv_block(x, filters=filters[0], kernel_size=3)
    x = conv_block(x, filters=filters[1], kernel_size=3, stride=2, num_groups=num_groups)
    x = conv_block(x, filters=filters[2], kernel_size=3, num_groups=num_groups)
    x = conv_block(x, filters=filters[3], kernel_size=3, num_groups=num_groups)
    x = GlobalAveragePooling2D()(x)
    x = Conv2D(1, kernel_size=1, padding='same')(x) # TODO: check padding

    return Model(inputs=[input], outputs=[x], name='discriminator')




###############################################################################
############################## GAN Model ######################################
###############################################################################

class FlatnetGAN(Model):
    def __init__(self, discriminator, generator):
        super(FlatnetGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        # losses
        self.d_loss = lambda d_true, d_fake : tf.math.log(d_true) - tf.math.log(1 - d_fake)


    def compile(self, d_optimizer, g_optimizer, g_perceptual_loss, adv_weight, mse_weight, perc_weight):
        super(FlatnetGAN, self).compile()
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
    







