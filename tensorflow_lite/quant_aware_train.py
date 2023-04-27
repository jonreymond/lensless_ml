import setGPU
import hydra
import tensorflow_model_optimization as tfmot
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ReduceLROnPlateau

                  

epochs = 30
batch_size = 16 * 2 * 2
test_size = 0.15
validation_size = 0.15
seed = 24

@hydra.main(version_base=None, config_path="configs", config_name="flatnet_reconstruction")
def main(conf):
    config = conf['reconstruction']
    

    model = tfmot.quantization.keras.quantize_model(model)
    tf.saved_model.save(model, 'aa')




if __name__ == '__main__':
    main()
    
    

    