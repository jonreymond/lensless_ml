# #############################################################################
# tf_inference.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tflite_runtime.interpreter as tflite
import numpy as np
import time
import sys
import hydra
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tf_dataset import *
from utils import psnr, ssim
import matplotlib.pyplot as plt
import cv2
import glob

from keras.losses import MeanSquaredError
mse = MeanSquaredError()

from sklearn import preprocessing


def get_from_folder(folder_path):
    measure_paths = glob.glob(folder_path + '/diffuser/*')
    groundtruth_paths = glob.glob(folder_path + '/lensed/*')
    measures = [np.expand_dims(np.load(path), 0) for path in measure_paths]
    groundtruths = [np.expand_dims(np.load(path), 0) *2 -1 for path in groundtruth_paths]
    return list(zip(measures, groundtruths))




def val_gen(reconstruct_config, num_samples=10):
    '''TODO
    '''
    dataset_config = reconstruct_config['dataset']
    indexes = np.arange(dataset_config['len'])    

    train_indexes, val_indexes = train_test_split(indexes, 
                                                        test_size=reconstruct_config['validation_split'], 
                                                        shuffle=True,
                                                        random_state=reconstruct_config['seed'])
    indexes = val_indexes
    num_samples = num_samples if num_samples else len(indexes)
    indexes = indexes[:num_samples]
    
    # Data Generator : set to 1 the batch size, as we want to evaluate each sample individually
    data_args = dict(batch_size=1, 
                    greyscale=reconstruct_config['greyscale'],
                    use_crop=reconstruct_config['use_crop'], 
                    seed=reconstruct_config['seed'])
    
    generator = get_tf_dataset(reconstruct_config['dataset']['name'], dataset_config, indexes, data_args).get()

    return ((tf.dtypes.cast(data, tf.float32), ground_truth.numpy()) for data, ground_truth in generator.take(num_samples))
    


def get_tflite_output(interpreter_dict, test_sample, print_time=False):
    '''From the tfliter interpreter dict generated from :py:func:create_tflite_interpreter, 
       evaluates the test sample's output of the tflite model
    Args:
        interpreter_dict (dict): the tflite interpreter, as well as the input, output pointers
        test_sample (numpy array): test sample to be evaluated
    Returns:
        numpy array: output of the tflite model
    '''
    interpreter = interpreter_dict['interpreter']
    interpreter.set_tensor(interpreter_dict['input'][0]['index'], test_sample)
    start_time = time.time()
    interpreter.invoke()
    runtime = (time.time() - start_time) * 1000
    if print_time:
        print('time: {:.3f} ms'.format(runtime))

    return interpreter.get_tensor(interpreter_dict['output'][0]['index'])


def get_interpreter_dict(tflite_path, num_threads=1):
    interpreter = tflite.Interpreter(model_path=tflite_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details = []
    for input in interpreter.get_input_details():  
        input_details.append(input)
    output_details = []
    for output in interpreter.get_output_details():
        output_details.append(output)
    return {'interpreter': interpreter, 'input':input_details, 'output':output_details}


def to_plottable_output(y):
        y = y[0]
        # y = (np.clip(y, -1, 1) + 1 ) / 2
        # y = np.clip(y, 0, 1)
        low_h = 0 
        high_h = 210
        low_w = 62
        high_w = 442
        
        y = np.array(y)
        
        print(y.shape)
        y = y[low_h:high_h, low_w:high_w]
        print(y.shape)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        y =np.flipud(y)

        return (y / y.max())[:,:,::-1]


@hydra.main(version_base=None, config_path="configs", config_name="tflite_inference")
def main(config):

    # tflite_camera_inversion = get_interpreter_dict(config['tflite_camera_inversion'], num_threads=config['num_threads'])
    # tflite_perceptual_model = get_interpreter_dict(config['tflite_perceptual_model'], num_threads=config['num_threads'])

    # camera_inversion_model = tf.saved_model.load(config['camera_inversion_model'])
    # camera_inversion_model = tf.saved_model.load(config['camera_inversion_model'])
    # perceptual_model = tf.saved_model.load(config['perceptual_model'])

    val_generator = val_gen(config)

    folder_path = '/home/jreymond/lensless_ml/results/' + config['name_final_folder'] + '/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # for idx, (sample, ground_truth) in enumerate(val_generator):
    for idx, (sample, ground_truth) in enumerate(get_from_folder(config['sample_images_folder'])):
        # intermediate_output = get_tflite_output(tflite_camera_inversion, sample)
        # intermediate_output = camera_inversion_model(sample)
        # output = intermediate_output
        # # output = get_tflite_output(tflite_perceptual_model, intermediate_output)
        # intermediate_output = sample
        # print(intermediate_output.shape)
        # # output = perceptual_model(intermediate_output)
        # # output = (output - np.min(output)) / (np.max(output) - np.min(output))
        # output = to_plottable_output(output)
        # # print(type(output), output.shape, output.dtype, output.max(), output.min())
        
        # plt.imsave(folder_path + str(idx) +'_output.png', output)
        # cv2.imwrite('output.png', output[0])
        # print(type(ground_truth), ground_truth.shape, ground_truth.dtype, ground_truth.max(), ground_truth.min())
        # ground_truth = np.flipud((ground_truth[0] + 1) / 2)
        ground_truth = to_plottable_output(ground_truth)
        plt.imsave(folder_path + str(idx) +'_ground.png', ground_truth)
        # print(idx)







        

    



if __name__ == "__main__":
    main()
    print('done')

    




