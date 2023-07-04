# #############################################################################
# tf_to_tflite.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################


import hydra
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import shutil
import pathlib
import numpy as np
import tensorflow as tf
import itertools
from collections import Counter
from tf_dataset import *
import sys
import shutil
import keras
from sklearn.model_selection import train_test_split
  
  
def representative_data_gen(reconstruct_config, camera_inversion=None, get_train=True, num_samples=None):
    '''TODO
    '''
    dataset_config = reconstruct_config['dataset']
    indexes = np.arange(dataset_config['len'])    


    train_indexes, val_indexes = train_test_split(indexes, 
                                                        test_size=reconstruct_config['validation_split'], 
                                                        shuffle=True,
                                                        random_state=reconstruct_config['seed'])
    indexes = train_indexes if get_train else val_indexes
    num_samples = num_samples if num_samples else len(indexes)
    indexes = indexes[:num_samples]
    
    # Data Generator : set to 1 the batch size, as we want to evaluate each sample individually
    data_args = dict(batch_size=1, 
                    greyscale=reconstruct_config['greyscale'],
                    use_crop=reconstruct_config['use_crop'], 
                    seed=reconstruct_config['seed'])
    
    generator = get_tf_dataset(reconstruct_config['dataset']['name'], dataset_config, indexes, data_args).get()
    
    # TODO: check if need to reshape
    # for x in generator:
    #     print(x[0].shape)
    #     break
    # return ([np.array(x[0], dtype=np.float32)] for x in generator)
    if camera_inversion:
        return ([tf.dtypes.cast(camera_inversion(data), tf.float32)] for data, ground_truth in generator.take(num_samples))
    else:
        return ([tf.dtypes.cast(data, tf.float32)] for data, ground_truth in generator.take(num_samples))



def create_tflite_interpreter(export_dir, with_optimization, with_quantization, store_path=None, repr_data_gen=None, use_debug=False,
                              num_threads=1, use_gpu=False):
    '''Creates the tflite interpreter from a tensorflow model
    Args:
        export_dir (str): location of the trained tensorflow model
        with_optimization (bool): if use optimization.default when creating the model
        with_quantization (bool): when optimization set to true, will quantize w.r.t. the representative dataset
        store_path (str, optional): Path where to store the .tflite model, if None will not store it. Defaults to None.
        repr_data_gen (generator, optional):  the representative dataset generator. Defaults to None.
    Returns:
        dict: the tflite interpreter, as well as the input, output pointers
    '''
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    
    if with_optimization:
        print('with weights compression')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
    if with_quantization:
        print('with only unit8 computation')
        if repr_data_gen:
            converter.representative_dataset = lambda: repr_data_gen
        

        # TFLITE_BUILTINS_INT8: Convert model using only TensorFlow Lite quantized int8 operations.
        # TFLITE_BUILTINS : Convert model using TensorFlow Lite builtin ops.
        # SELECT_TF_OPS : with tensorflow ops

        converter.target_spec.supported_ops = [ 
                                                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                                # tf.lite.OpsSet.TFLITE_BUILTINS,
                                                # tf.lite.OpsSet.SELECT_TF_OPS
                                                  ] 
        
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8
        
    tflite_model = converter.convert()

    debugger = None
    if use_debug:
        debugger = tf.lite.experimental.QuantizationDebugger(
                        converter=converter, debug_dataset=(lambda : repr_data_gen))
    if store_path is not None:
        tflite_model_file = pathlib.Path(store_path)
        tflite_model_file.write_bytes(tflite_model)

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    #multiple inputs
    input_details = []
    for input in interpreter.get_input_details():  
        input_details.append(input)
    output_details = []
    for output in interpreter.get_output_details():
        output_details.append(output)

    return {'interpreter': interpreter, 'input':input_details, 'output':output_details}, debugger


def get_tflite_output(interpreter_dict, test_sample):
    '''From the tfliter interpreter dict generated from :py:func:create_tflite_interpreter, 
       evaluates the test sample's output of the tflite model
    Args:
        interpreter_dict (dict): the tflite interpreter, as well as the input, output pointers
        test_sample (numpy array): test sample to be evaluated
    Returns:
        numpy array: output of the tflite model
    '''
    interpreter = interpreter_dict['interpreter']
    interpreter.set_tensor(interpreter_dict['input'], test_sample)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter_dict['output'])


def tflite_to_cc(name_model, tflite_path):
    '''Transfrom a .tflite file to a .cc + .h file. It assumes that the .tflite is stored in config['tflite_path'].
        it will store the result in the same folder where the .tflite is located
        
    '''
    path_cc = tflite_path.replace(".tflite", "_temp.cc")
    os.system("xxd -i " + tflite_path + " > " + path_cc)

    old_name_model = tflite_path.replace('/', '_').replace('.', '_')

    outfile = path_cc.replace("_temp.cc", ".cc")

    #write final .cc file
    with open(path_cc) as fin, open(outfile, "w+") as fout:
        fout.write( '''#include "'''+ name_model + '.h' + '''"'''+ '\n\n')
        for line in fin:
            new_line = line
            new_line = new_line.replace(old_name_model, name_model)
            new_line = new_line.replace('unsigned char', 'alignas(16) const unsigned char')
            new_line = new_line.replace('unsigned int', 'const unsigned int')
            fout.write(new_line)

    #write final .h file
    outfile_h = outfile.replace(".cc", ".h")
    with open(outfile_h, "w+") as fout:
        fout.write("#include <cstdint>" '\n\n')
        fout.write("extern const unsigned char " + name_model + "[];\n")
        fout.write("extern const unsigned int " + name_model + "_len;\n")

    #remove temporary file
    os.remove(path_cc)





@hydra.main(version_base=None, config_path="configs", config_name="tflite_conversion")
def main(config):

    print('='* 80)

    num_samples = config['num_samples'] if config['num_samples'] != 'None' else None
    
    
    camera_inversion_model_path = os.path.join(config['store_folder'], 'tensorflow', 'models', 'camera_inversion.pb')
    camera_inversion_model = None
    if os.path.exists(camera_inversion_model_path):
        camera_inversion_model = keras.models.load_model(camera_inversion_model_path)

    if config['representative_dataset']:
        repr_data_gen = representative_data_gen(config, num_samples=num_samples, camera_inversion=camera_inversion_model)
    else:
        repr_data_gen = None


    perceptual_model_path = os.path.join(config['store_folder'], 'tensorflow', 'models', 'perceptual_model.pb')

    tflite_folder = os.path.join(config['store_folder'], 'tflite')
    name_model = config['tflite_model_name']
    tflite_path = os.path.join(tflite_folder, name_model + '.tflite')
    if not os.path.exists(os.path.dirname(tflite_path)):
        os.makedirs(os.path.dirname(tflite_path))


    # model = keras.models.load_model(model_path)
    # print(model.summary())
    # print(model.perceptual_model)
    # print(model.perceptual_model.summary())

    
    tflite_interp_quant = create_tflite_interpreter(perceptual_model_path, 
                                                    with_optimization=config['with_optimization'], 
                                                    with_quantization=config['with_quantization'], 
                                                    store_path=tflite_path,
                                                    repr_data_gen= repr_data_gen,
                                                    use_debug=config['use_debug']
                                                    )

    if config['with_cc_files']:
        tflite_to_cc(name_model=name_model,
                    tflite_path=tflite_path)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    temp_store_folder = hydra_cfg['runtime']['output_dir']
    if os.path.exists(os.path.join(tflite_folder, '.hydra')):
        shutil.rmtree(os.path.join(tflite_folder, '.hydra'), ignore_errors=True)

    shutil.move(os.path.join(temp_store_folder, '.hydra'), 
              os.path.join(tflite_folder, ''))
    os.rename(os.path.join(temp_store_folder, 'tf_to_tflite.log'),
              os.path.join(tflite_folder, 'tf_to_tflite.log'))
    os.rmdir(temp_store_folder)






if __name__ == "__main__":
    main()
    print('done')