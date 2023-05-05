from keras import regularizers

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers import BatchNormalization, UpSampling2D, Concatenate, Input, Conv2DTranspose, Resizing, ZeroPadding2D, Lambda

from keras.models import Model
import tensorflow as tf
from utils import *



def conv_block(x, filters, kernel_size, strides=1, bn_eps=1e-3, l1_factor=0, l2_factor=0):
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

    # axis=1 for NCHW
    x = BatchNormalization(epsilon=bn_eps, axis=1)(x)
    x = Activation('relu')(x)
    return x


def stack_encoder(x, filters, kernel_size=(3, 3), bn_eps=1e-3):
    x = conv_block(x, filters, kernel_size, bn_eps=bn_eps)
    x = conv_block(x, filters, kernel_size, bn_eps=bn_eps)
    down_tensor = x
    x_small = MaxPooling2D(pool_size=2, strides=2)(x)
    return x_small, down_tensor


def stack_decoder(x, filters, down_tensor, kernel_size=3, bilinear=True, bn_eps=1e-3, num_conv=2):
    height, width = down_tensor.shape[2:]

    if bilinear:
        #  Exact upsampling
        # x = to_channel_last(x)
        # x = Resizing(height, width,interpolation='bilinear')(x)
        # x = to_channel_first(x)
        x = UpSampling2D(size=2, interpolation="bilinear")(x)
        x = ZeroPadding2D(((0, down_tensor.shape[2] - x.shape[2]), (0, down_tensor.shape[3] - x.shape[3])))(x)

    else:
        raise NotImplementedError
        # Transposed convolution 
        # x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
        # y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)


    x = Concatenate(axis=1)([x, down_tensor])
    # decode : TODO : normally only 2
    for i in range(num_conv):
        x = conv_block(x, filters, kernel_size, bn_eps=bn_eps)

    return x



def u_net(input, enc_filters, name='unet', last_conv_filter=None, num_dec_conv=2, bn_eps=1e-3, out_shape=None):
    x = input
    ### down: encoder ###

    down_tensors = []
    for enc_filter in enc_filters:
        x, down_tensor = stack_encoder(x, enc_filter, kernel_size=3, bn_eps=bn_eps)
        down_tensors.append(down_tensor)
    

    ### Center ###
    x = conv_block(x, filters=enc_filters[-1], kernel_size=3, bn_eps=bn_eps)
    

    ### up: decoder ###
    down_tensors = down_tensors[::-1]
    dec_filters = enc_filters[::-1]
    dec_filters = dec_filters[1:] +[dec_filters[-1]]

    for dec_filter, down_tensor in zip(dec_filters, down_tensors):
        x = stack_decoder(x, dec_filter, down_tensor, kernel_size=3, bn_eps=bn_eps, num_conv=num_dec_conv)

    if last_conv_filter:
        x = conv_block(x, last_conv_filter, kernel_size=3)
    ### "classifier" ###
    # TODO : check if input.shape[1] == input_shape[0]
    num_outputs = 3 if input.shape[1] != 1 else 1

    x = Conv2D(filters=num_outputs, kernel_size=1, use_bias=True, padding='same')(x)

    if out_shape:
        # Exact resizing without trainable parameters
        x = to_channel_last(x)
        # works, but not quantized
        x = Lambda(lambda x: tf.image.resize(x, size=out_shape[1:3], method=tf.image.ResizeMethod.BILINEAR))(x)
        # x = tf.keras.layers.Resizing(height=out_shape[1], width=out_shape[2], interpolation='bilinear')(x)
        x = to_channel_first(x)

        # for QAT, need to 

    return x

 
    