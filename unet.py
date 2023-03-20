# import torch
# import torch.nn as nn
# import torch.nn.functional as F

from keras import regularizers

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers import BatchNormalization, UpSampling2D, Concatenate, Input

from keras.models import Model




def conv_block(x, filters, kernel_size, strides=1, l1_factor=0, l2_factor=0):
    '''Convolution block : convolution --> batchnormalization --> relu
    Args:
        x (input): input
        filters (int): number of filters
        kernel_size (int): kernel size
        strides (int, optional): strides number. Defaults to 1.
        l1_factor (int, optional): l1 regulation factor. Defaults to 0.
        l2_factor (int, optional): l2 regulation factor. Defaults to 0.
    Returns:
        output: input passed through convolution block
    '''
    
    x = Conv2D(filters,
                kernel_size=kernel_size,
                strides=strides,
                padding='same', # TODO: check
                kernel_initializer='glorot_uniform'
                ,kernel_regularizer=regularizers.L1L2(l1=l1_factor, l2=l2_factor)
                )(x)

    x = BatchNormalization(epsilon=1e-4)(x)
    x = Activation('relu')(x)
    return x


def stack_encoder(x, filters, kernel_size=(3, 3)):
    # padding = (kernel_size - 1) // 2
    x = conv_block(x, filters, kernel_size)
    x = conv_block(x, filters, kernel_size)
    x_small = MaxPooling2D(pool_size=2, strides=2)(x)
    return x, x_small


def stack_decoder(x, filters, down_tensor, kernel_size=3):
    # TODO : Check
    height, width = down_tensor.shape[2:]
    x = UpSampling2D(size=(height, width), interpolation='bilinear')(x)
    x = Concatenate(axis=-1)([x, down_tensor])
    # decode
    for i in range(3):
        x = conv_block(x, filters, kernel_size)
    return x




def u_net(shape):
    input = Input(shape=shape, name="input")
    x = input

    # down: encoder
    enc_filters = [24, 64, 128, 256, 512]
    for enc_filter in enc_filters:
        x = stack_encoder(x, enc_filter, kernel_size=3)

    # Center
    # TODO : check padding
    x = conv_block(x, filters=enc_filters[-1], kernel_size=1)

    # up: decoder
    dec_filters = enc_filters[::-1]
    down_tensors = dec_filters[1:] + [dec_filters[-1]]
    for dec_filter, down_tensor in zip(dec_filters, down_tensors):
        x = stack_decoder(x, dec_filter, down_tensor, kernel_size=3)

    # classifier
    x = Conv2D(filters=3, kernel_size=3, use_bias=True)(x)

    return Model(inputs=[input], outputs=[x], name='u_net')

 
    