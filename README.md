# Lensless imaging, deep learning, and deploymnent

## Structure
- configs: 
  - dataset : this folder contains all the parameters describing the dataset you want to use, it contains the parameters of the DiffuserCam (wallerlab), FlatNet and PhlatNet dataset
  - model : this folder contains all the default models that could be chosen for the perceptual model (the U-Net and its variations). You can define here your custom U-Net based on the example provided
  - tflite_conversion : this file contains all the information for converting a TensorFlow model to a TfLite model, it is called when we run the script tf_to_tflite.py
  - tflite_inference : this file contains all the information for running an tflite inference in the python script tflite_inference.py
  - train_reconstruction : this file contains all the information for training a model in train.py
- tflite: contains tools for visualize the tflite model
- benchmark: script to run a TensorFlow model
- callbacks: contains all the callback methods used in the training
- camera_inversion_numpy : contains the numpy version of the camera inversion layer if we want to use them in an embedded system
- model : contains the model implementations
- tf_dataset : contains the dataloader for DiffuserCam, FlatNet and PhlatNet. The implementation for FlatNet uses the already processed dataset already downsampled and separated in 4 channels and stored in numpy.
- tf_to_tflite : script to convert a tf model to a tflite model
- tflite_inference : script to run an inference of a tflite model
- train : script to train the model
- utils : utilitary methods, mostly used during the training

## Remarks
- To use the Tensoboard visualization, you have to run "tensorboard --logdir 'path to tensoboard folder'"
- The binary files to get the benchmark of Tlite models can be found in this [link](https://www.tensorflow.org/lite/performance/measurement)
- For visualizing the tflite model, run : "python -m tflite_visualize 'model'.tflite visualized_model.html". It will generate an html file. This can be useful to see which operators are used in the tflite model.
