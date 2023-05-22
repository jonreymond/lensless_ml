import torch
import onnx
import os
import onnxsim
# from torch.utils.tensorboard import SummaryWriter
from onnx_tf.backend import prepare
import tensorflow as tf


def to_tf_graph(torch_model, sample_input, store_output):
    print('exporting torch to onnx model...')
    torch_model.eval()

    # torch_out = torch_model(sample_input)    

    torch.onnx.export(model = torch_model,               # model being run
                  args = sample_input,                         # model input (or a tuple for multiple inputs)
                  f = store_output,   # where to save the model (can be a file or file-like object)
                  export_params = True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  verbose=True,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input1', 'input2'],   # the model's input names
                  output_names = ['output'], # the model's output names
                #   training=torch.onnx.TrainingMode.TRAINING,
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    
    print('onnx model saved to ', store_output)
    onnx_model = onnx.load(store_output)

    # load and change dimensions to be dynamic
    for dim in (0, 2, 3):
        onnx_model.graph.input[0].type.tensor_type.shape.dim[dim].dim_param = '?'
        onnx_model.graph.input[1].type.tensor_type.shape.dim[dim].dim_param = '?'

    # onnx_model, success = onnxsim.simplify(store_output)
    # assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
    # simple_onnx_model_path =  f'{store_output}.simple.onnx'
    # print(f'Generating {simple_onnx_model_path} ...')
    # onnx.save(onnx_model, simple_onnx_model_path)

    # writer.add_onnx_graph(store_output)
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))

    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(store_output + '.pb') 

    return tf.keras.models.load_model(store_output + '.pb')


