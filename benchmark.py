import os
import numpy as np
import time
import sys

# with inversion
# model_path = '/root/jreymond/lensless_ml/outputs/2023-06-06/06-56-56/tensorflow/models/wallerlab_unet.pb'
# without_inversion
model_path = '/root/jreymond/lensless_ml/outputs/2023-06-02/10-40-15/tensorflow/models/wallerlab_unet.pb'

use_gpu = False


if use_gpu:
    # set gpu to the less used one
    import setGPU
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import keras





if __name__ == "__main__":

    model = tf.saved_model.load(model_path)

    # shape: num_samples=1, height=270, width=480, channels=3
    shape = (1, 270, 480, 3)
    # should be between [-1, 1]
    test_sample = np.random.rand(*shape).astype(np.float32)
    # warmup, output shape = (1, 210, 380, 3)
    # output : between [-1, 1]
    output = model(test_sample)
    
    for i in range(10):
        test_sample = np.random.rand(*shape).astype(np.float32) * 2 - 1
        start_time = time.time()
        output = model(test_sample)
        runtime = (time.time() - start_time) * 1000
        print('time: {:.3f} ms'.format(runtime))
