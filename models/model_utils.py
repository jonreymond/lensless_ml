from models.unet import *
from models.flatnet.camera_inversion import *


    

def get_model(config, input_shape, out_shape, model_name='Reconstruction model'):
    model_config = dict(config['model'][config['model_name']])
    input = Input(shape=input_shape, name='Input')
    x = input

    if config['use_camera_inversion']:
        x = get_camera_inversion_layer(x, config)

    if model_config['type'] == 'unet':
        model_config.pop('type')
        x = u_net(input=x, **model_config, out_shape=out_shape)(x)
        
    else:
        raise ValueError(f'Unknown model type: {model_config["type"]}')
    
    return Model(inputs=[input], outputs=[x], name=model_name)


