# #############################################################################
# benchmark.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

import os
import numpy as np
import time
import sys

# with inversion
# model_path = '/root/jreymond/lensless_ml/outputs/2023-06-06/06-56-56/tensorflow/models/wallerlab_unet.pb'
model_path = '/home/jreymond/lensless_ml/temporary/gen_pruned.pb'
# model_path = '/home/jreymond/lensless_ml/outputs/2023-06-23/07-02-26/tensorflow/models/gen_unet64.pb'
# without_inversion
# model_path = '/root/jreymond/lensless_ml/outputs/2023-06-02/10-40-15/tensorflow/models/wallerlab_unet.pb'

use_gpu = True


if use_gpu:
    # set gpu to the less used one
    import setGPU
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import keras





if __name__ == "__main__":

    # model = tf.saved_model.load(model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # shape: num_samples=1, height=270, width=480, channels=3
    shape = (1, 270, 480, 3)
    # should be between [-1, 1]
    test_sample = np.random.rand(*shape).astype(np.float32)
    # warmup, output shape = (1, 210, 380, 3)
    # output : between [-1, 1]
    output = model(test_sample)
    
    res = []
    for i in range(50):
        test_sample = np.random.rand(*shape).astype(np.float32) * 2 - 1
        start_time = time.time()
        output = model(test_sample)
        runtime = (time.time() - start_time) * 1000
        res.append(runtime)
        print('time: {:.3f} ms'.format(runtime))
        time.sleep(1)
    print('mean: {:.3f} ms'.format(np.mean(res)))
