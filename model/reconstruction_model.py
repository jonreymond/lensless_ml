# #############################################################################
# reconstruction_model.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################


import tensorflow as tf

import keras
from keras import Model
from keras.utils.layer_utils import count_params
from keras.layers import Input

from .camera_inversion_layers import SeparableLayer, FTLayer, get_toeplitz_init
from .perceptual_models import u_net, experimental_models





#################################################################################################################
######################################## Reconstruction Model ###################################################
#################################################################################################################


class UnetModel(keras.Sequential):

    def __init__(self, 
                 input_shape, 
                 output_shape, 
                 psf=None,
                 phi_l=None,
                 phi_r=None,
                 perceptual_args=None,
                 camera_inversion_args=None, 
                 model_weights_path=None, 
                 name='reconstruction_model'):
        """Reconstruction model using a perceptual model and a camera inversion layer
        
        Args:
            input_shape (tuple): input shape
            output_shape (tuple): output shape
            psf (numpy array, optional): point-spread-function matrix, shape (H, W, C). Defaults to None.
            phi_l (numpy array, optional): left matrix for separable case. Defaults to None.
            phi_r (numpy array, optional): right matrix for separable case. Defaults to None.
            perceptual_args (dict, optional): arguments for the perceptual(U-Net) model. Defaults to None.
            camera_inversion_args (dict, optional): arguments for the camera inversion layer. Defaults to None."""
        
        super().__init__(name=name)

        self.in_shape = input_shape

        input = Input(shape=input_shape, name='input', dtype='float32')
        x = input
        # camera inversion layer #
        camera_inversion_layer = None
        if camera_inversion_args:
            if isinstance(camera_inversion_args, SeparableLayer) or isinstance(camera_inversion_args, FTLayer) or isinstance(camera_inversion_args, tf.keras.Model):
                camera_inversion_layer = camera_inversion_args

            else:
                # Separable dataset
                if camera_inversion_args['type'] == 'separable':
                    if phi_l is None or phi_r is None:
                        print('phi_l and phi_r not provided, using random init')
                        target_shape = camera_inversion_args['target_shape']
                        slope = camera_inversion_args['slope']
                        phi_l = get_toeplitz_init(target_shape, slope, is_left=True, seed=1)
                        phi_r = get_toeplitz_init(target_shape, slope, is_left=False, seed=1)

                    camera_inversion_layer = SeparableLayer(phi_l, phi_r)

                # Non separable
                elif camera_inversion_args['type'] == 'non_separable':
                    if psf is None:
                        raise NotImplementedError('PSF is None, random init (PSF) not implemented for non-separable dataset')
                    
                    camera_inversion_args =  dict(camera_inversion_args)
                    camera_inversion_args.pop('type')
                    camera_inversion_layer = FTLayer(psf=psf, **camera_inversion_args)
                else:
                    raise NotImplementedError('Camera inversion type not implemented, choose between separable and non_separable')

        self.camera_inversion_layer = camera_inversion_layer


        if camera_inversion_layer:
            x = camera_inversion_layer(input)
            
        if isinstance(perceptual_args, keras.Model):
            self.perceptual_model = perceptual_args
        
        else:
            model_config = dict(perceptual_args)
            print('model config', model_config)
            model_type = model_config.pop('type')

            if model_type == 'unet':
                model_output = [u_net(input=x, **model_config, out_shape=output_shape)]
            else:
                model_output = [experimental_models(model_name=model_config['model_name'], 
                                                    input=x, 
                                                    out_shape=output_shape,
                                                    model_args=model_config['args'])]

            self.perceptual_model = Model(inputs=[x],
                                        outputs=model_output,
                                        name='perceptual_model')
        
        layers = [input]
        if self.camera_inversion_layer:
            if isinstance(self.camera_inversion_layer, tf.keras.Model):
                self.camera_inversion_layer.build(input_shape=self.in_shape)

            layers.append(self.camera_inversion_layer)
        
        layers.append(self.perceptual_model)
        super().__init__(name=name, layers=layers)

        if model_weights_path:
            self.load_weights(model_weights_path).expect_partial()

    
    def summary(self, **kwargs):
        cam_model = None
        if self.camera_inversion_layer:
            inp = Input(shape=self.in_shape, name='input', dtype='float32')
            out = self.camera_inversion_layer(inp)
            cam_model = Model(inputs=[inp], outputs=[out])
            if isinstance(self.camera_inversion_layer, tf.keras.Model):
                self.camera_inversion_layer.summary(**kwargs)
            else:
                cam_model.summary(**kwargs)

        self.perceptual_model.summary(**kwargs)

        if cam_model:
            model = keras.Sequential([cam_model, self.perceptual_model])
            line_length = 98
            print("=" * line_length)
            trainable_count = count_params(model.trainable_weights)
            non_trainable_count = count_params(model.non_trainable_weights)

            print(f"Total params: {trainable_count + non_trainable_count:,}")
            print(f"Trainable params: {trainable_count:,}")
            print(f"Non-trainable params: {non_trainable_count:,}")
            print("_" * line_length)
        


