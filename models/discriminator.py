
from keras.models import Model
from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
import tensorflow as tf

#####################################################################
###################### Discriminator ################################
#####################################################################

# def conv_block(x, filters, kernel_size, strides=(1,1), activation='swish', use_batchnorm=False, num_groups=None):
#     x = Conv2D(filters,
#                 kernel_size=kernel_size,
#                 strides=strides,
#                 padding='same', # TODO : check
#                 )(x)
#     if num_groups:
#         x = GroupNormalization(groups=num_groups)(x)

    
#     x = Activation(activation=activation)(x)
#     return x
    
# # num_groups : for group normalization 
# # TODO : check if num_groups should = num_batches, or <
# def get_discriminator(shape, activation='swish', num_groups=8):
#     input = Input(shape=shape, name="input")
#     x = input
#     filters = [64, 128, 128, 256]
#     x = conv_block(x, filters=filters[0], kernel_size=3, activation=activation)
#     x = conv_block(x, filters=filters[1], kernel_size=3, strides=2, num_groups=num_groups, activation=activation)
#     x = conv_block(x, filters=filters[2], kernel_size=3, num_groups=num_groups, activation=activation)
#     x = conv_block(x, filters=filters[3], kernel_size=3, num_groups=num_groups, activation=activation)

#     x = GlobalAveragePooling2D(keepdims=True)(x)

#     x = Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(x) # TODO: check padding
#     x = Reshape(target_shape=[])(x)
#     # x = Activation("sigmoid")(x)
    
#     return Model(inputs=[input], outputs=[x], name='discriminator')



def get_discriminator(shape, filters, strides, kernel_size, activation='swish', use_groupnorm=False, num_groups=None, sigmoid_output=False):
    assert activation, "activation must be specified"
    input = Input(shape=shape, name="input")
    x = input

    assert len(filters) == len(strides) and len(strides) == len(kernel_size)

    for i in range(len(filters)):
        # conv block
        x = Conv2D(filters[i],
                kernel_size=kernel_size[i],
                strides=strides[i],
                padding='same', # TODO : check
                )(x)
        if use_groupnorm:
            x = GroupNormalization(groups=num_groups)(x)
        else:
            x = BatchNormalization()(x)
        
        x = Activation(activation=activation)(x)
        

    x = GlobalAveragePooling2D(keepdims=True)(x)

    x = Conv2D(1, kernel_size=1, padding='same', activation=None)(x) # TODO: check padding
    x = Reshape(target_shape=[])(x)

    if sigmoid_output:
        x = Activation("sigmoid")(x)
    
    return Model(inputs=[input], outputs=[x], name='discriminator')