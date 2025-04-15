# #############################################################################
# perceptual_models.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################




import tensorflow as tf

from keras import regularizers

from keras.layers import BatchNormalization, UpSampling2D, Concatenate, ZeroPadding2D, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation

from keras_unet_collection import models as M_unet



#################################################################################################################
######################################### Perceptual Model ######################################################
#################################################################################################################


######################################### Basic U-Net ##########################################################

def conv_block(x, 
               filters, 
               kernel_size, 
               strides=1, 
               bn_eps=1e-3, 
               l1_factor=0, 
               l2_factor=0, 
               padding='same'):
    '''Convolution block : 2D convolution --> batchnormalization --> relu

    Args:
        x (input): input
        filters (int): number of filters
        kernel_size (int): kernel size
        strides (int, optional): strides number. Defaults to 1.
        l1_factor (int, optional): l1 regulation factor. Defaults to 0.
        l2_factor (int, optional): l2 regulation factor. Defaults to 0.
        padding (str, optional): padding type. Defaults to 'same'.
    Returns:
        output: input passed through convolution block
    '''
    
    x = Conv2D(filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                kernel_initializer='glorot_uniform'
                ,kernel_regularizer=regularizers.L1L2(l1=l1_factor, l2=l2_factor)
                )(x)

    # axis=1 for NCHW
    x = BatchNormalization(epsilon=bn_eps)(x)
    x = Activation('relu')(x)
    return x


def stack_encoder(input, filters, kernel_sizes=[3, 3], bn_eps=1e-3, maxpool=True, intermediate_nodes=False):
    if not maxpool:
        padding = ['valid', 'same']
        strides = [2, 1]
    else:
        padding = ['same', 'same']
        strides = [1, 1]
    x = conv_block(input, filters, kernel_sizes[0], bn_eps=bn_eps, strides=strides[0], padding=padding[0])
    x = conv_block(x, filters, kernel_sizes[1], bn_eps=bn_eps, strides=strides[1], padding=padding[1])
    down_tensor = x
    if maxpool:
        x_small = MaxPooling2D(pool_size=2, strides=2)(x)
        if intermediate_nodes:
            down_tensor = conv_block(x_small, filters, kernel_size=1, bn_eps=bn_eps, strides=1, padding='same')
        return x_small, down_tensor
    else:
        if intermediate_nodes:
            input = conv_block(input, filters, kernel_size=1, bn_eps=bn_eps, strides=1, padding='same')
        return down_tensor, input

    


def stack_decoder(x, filters, down_tensor, kernel_size=3, bilinear=True, bn_eps=1e-3, num_conv=2):
    height, width = down_tensor.shape[2:]

    if bilinear:
        #  Exact upsampling
        # x = to_channel_last(x)
        # x = Resizing(height, width,interpolation='bilinear')(x)
        # x = to_channel_first(x)
        x = UpSampling2D(size=2, interpolation="bilinear")(x)
        x = ZeroPadding2D(((0, down_tensor.shape[1] - x.shape[1]), (0, down_tensor.shape[2] - x.shape[2])))(x)

    else:
        raise NotImplementedError
        # Transposed convolution 
        # x = Conv2DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
        # y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)

    # Concatenate in NHWC
    x = Concatenate(axis=-1)([x, down_tensor])
    # decode : TODO : normally only 2
    for i in range(num_conv):
        x = conv_block(x, filters, kernel_size, bn_eps=bn_eps)

    return x


def resize_input(input, factor):
    while input % factor != 0:
        input += 1
    return input


def u_net(input, enc_filters, maxpool=True, intermediate_nodes=False, first_kernel_size=3,
          last_conv_filter=None, num_dec_conv=2, bn_eps=1e-3, out_shape=None, output_activation='tanh',
          depth_space=False):
    
    x = input
    if depth_space:
        factor = 2
        # height, width = input.shape[1:3]
        # new_height = resize_input(height, factor=factor)
        # new_width = resize_input(width, factor=factor)
        # print('new_height', new_height, 'new_width', new_width)

        # x = tf.keras.layers.Resizing(new_height, new_width)(input)
        x = tf.nn.space_to_depth(x, factor)


    ### down: encoder ###

    down_tensors = []

    if not maxpool:
        x = conv_block(x, enc_filters[0], kernel_size=3, bn_eps=bn_eps, strides=1)
        x = conv_block(x, enc_filters[0], kernel_size=3, bn_eps=bn_eps, strides=1)

    kernel_sizes = [[3,3]] * len(enc_filters)
    if not first_kernel_size:
        first_kernel_size = 3
    kernel_sizes[0] = [first_kernel_size,3]

    
    for i in range(len(enc_filters)):
        x, down_tensor = stack_encoder(x, enc_filters[i], kernel_sizes=kernel_sizes[i], bn_eps=bn_eps, maxpool=maxpool, intermediate_nodes=intermediate_nodes)
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
    num_outputs = 3 if input.shape[1] != 1 else 1
    if depth_space:
        num_outputs *= 2**2
    x = Conv2D(filters=num_outputs, kernel_size=1, use_bias=True, padding='same', activation=output_activation)(x)

    if depth_space:
        x = tf.nn.depth_to_space(x, 2)

    if out_shape:
        # works, but not quantized
        size = out_shape[0:2]
        x = Lambda(lambda x: tf.image.resize(x, size=size, method=tf.image.ResizeMethod.BILINEAR))(x)

    return x



####################################### U-Net variation models ##################################################

MODELS = dict(
            unet_2d = M_unet.unet_2d,
            vnet_2d = M_unet.vnet_2d,
            unet_plus_2d = M_unet.unet_plus_2d,
            r2_unet_2d = M_unet.r2_unet_2d,
            att_unet_2d = M_unet.att_unet_2d,
            resunet_a_2d = M_unet.resunet_a_2d,
            u2net_2d = M_unet.u2net_2d,
            unet_3plus_2d = M_unet.unet_3plus_2d,
            transunet_2d = M_unet.transunet_2d,
            swin_unet_2d = M_unet.swin_unet_2d)


def resize_input(dim, unet_depth):
    while dim % 2**(unet_depth + 1) != 0:
        dim += 1
    return dim


def experimental_models(model_name, input, out_shape, model_args):
    model_args = dict(model_args)

    output_activation = model_args.pop('output_activation')
    model_args['output_activation'] = None

    _, height, width, channels = input.shape

    if 'filter_num' in model_args:
        unet_depth = len(model_args['filter_num'])
        

    elif 'filter_num_down' in model_args:
        unet_depth = len(model_args['filter_num_down'])

    elif 'depth' in model_args:
        unet_depth = model_args['depth']
        
    new_height = resize_input(height, unet_depth)
    new_width = resize_input(width, unet_depth)

    if model_name in ['transunet_2d', 'swin_unet_2d', 'unet_3plus_2d']:
        new_height = max(resize_input(height, unet_depth), resize_input(width, unet_depth))
        new_width = new_height

    x = tf.keras.layers.Resizing(new_height, new_width)(input)
    
    model_args['n_labels'] = channels
    model_args['input_size'] = (new_height, new_width, channels)
    model_args['name'] = model_name
    
    gen_model = MODELS[model_name](**model_args)
    gen_model.summary()

    x = gen_model(x)

    print('output shape: ')
    if output_activation:
        x = Activation(output_activation)(x)

    x = tf.keras.layers.Resizing(height=out_shape[0], width=out_shape[1])(x)

    return x
