import tflite_runtime.interpreter as tflite
import numpy as np
import time
import sys


model_file = 'tflite/unet128.tflite'



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
    interpreter.set_tensor(interpreter_dict['input'][0]['index'], test_sample)
    start_time = time.time()
    interpreter.invoke()
    runtime = (time.time() - start_time) * 1000
    print('time: {:.3f} ms'.format(runtime))

    return interpreter.get_tensor(interpreter_dict['output'][0]['index'])


def get_interpreter_dict(tflite_path):
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = []
    for input in interpreter.get_input_details():  
        input_details.append(input)
    output_details = []
    for output in interpreter.get_output_details():
        output_details.append(output)
    return {'interpreter': interpreter, 'input':input_details, 'output':output_details}



if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_file = sys.argv[1]

    interpreter_dict = get_interpreter_dict(model_file)
    
    # shape: num_batch=1, height=304, width=432, channels=4 (bayer)
    # should be between [-1, 1]
    test_sample = np.random.rand(1, 304, 432, 4).astype(np.float32)

    output = get_tflite_output(interpreter_dict, test_sample)

    #output : between [-1, 1]
    print(output.shape, output.dtype)

    print('done')




