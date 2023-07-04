# #############################################################################
# discriminator.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################


from keras.models import Model
from keras.layers import GroupNormalization, Input, GlobalAveragePooling2D, Reshape, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
import tensorflow as tf

#####################################################################
###################### Discriminator ################################
#####################################################################


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