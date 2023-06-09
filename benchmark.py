import os
import numpy as np
import time
import sys

model_path = ''
use_gpu = False


if use_gpu:
    # set gpu to the less used one
    import setGPU
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import keras





if __name__ == "__main__":


    model = keras.models.load_model(model_path)

    test_sample = np.random.rand(1, 304, 432, 4).astype(np.float32)

    start_time = time.time()
    output = model(test_sample)
    runtime = (time.time() - start_time) * 1000
    print('time: {:.3f} ms'.format(runtime))
