# Lensless imaging, deep learning, and deploymnent
## Description
Lensless imaging provides a large panel of benefits : cost, size, weight, etc., that are crucial for wearable applications, IoT or medical devices. Such setups require the design of reconstruction algorithms to recover the image from the captured measurements. Most of the current SoTA recon- struction models use deep learning, but the results provided are hardly reproducible and mostly not meant to be deployed into embedded systems.

In this work, we implement the work of [Monakhova et al.](https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-20-28075&id=420747) that uses uniquely deep learning, and the work of by [Khan et al](https://arxiv.org/pdf/2010.15440). We then present a way to transform these models to be deployable using TensorFlow Lite, and evaluate the benefits of model optimization techniques such as quantization-aware training(QAT), weight pruning, or weight clustering.

This work is part of a master thesis, where the report can be found [here in the EPFL website](https://infoscience.epfl.ch/entities/publication/50c03138-7854-4097-a9c3-c77571411a65), or in the **`docs/`** folder.

## Project Structure
- **`configs/`**: Stores all configuration files used for different aspects of the project, including datasets, models, and training.
  - **`dataset`** : contains all the parameters describing the dataset you want to use, it contains the parameters of the DiffuserCam (wallerlab), FlatNet and PhlatNet dataset
  - **`model`** : contains all the default models that could be chosen for the perceptual model (the U-Net and its variations). You can define here your custom U-Net based on the example provided
  - **`tflite_conversion`** : this file contains all the information for converting a TensorFlow model to a TfLite model, it is called when we run the script tf_to_tflite.py
  - **`tflite_inference`** : this file contains all the information for running an tflite inference in the python script tflite_inference.py
  - **`train_reconstruction`** : this file contains all the information for training a model in train.py  

- **`docs/`**: Contains the presentation and the report

- **`environments/`**: Contains environment setup files for different CONDA setups.**`lensless_ml.yml`** is the default conda environment configuration.
- **`example.ipynb`**: Jupyter notebook providing an example of the results we could have using the models
- **`tf_inference`**:script to load and run a TensorFlow model 
- **`tf_to_tflite.py`**: Script for converting TensorFlow models to TFLite format.
- **`tflite_inference.py`**: Script for running inference on TFLite models.
- **`train.py`**: Main training script used to train the models with specified configurations.
- **`utils/`**: Contains utility functions used across different scripts in the project.
- **`tflite/`**: Contains tools for visualizing TFLite models and visual outputs.

- **`model/`**:
  - **`__init__.py`**: Marks the directory as a Python package.
  - **`callbacks.py`**: Contains training callbacks, such as early stopping or learning rate schedulers during the training.
  - **`camera_inversion_layers.py`**: Defines layers related to camera inversion in the model, separable and non-separable.
  - **`gan_model.py`**: Contains the GAN (Generative Adversarial Network) model and related components.
  - **`perceptual_models.py`**: Defines perceptual models, mainly U-Nets and advanced types of U-Nets.
  - **`quantization.py`**: Defines methods for model quantization.
  - **`reconstruction_model.py`**: Defines the final model, having a camera inversion layer and a chosen perceptual model.
  - **`tf_dataset.py`**: Contains code for loading and preprocessing datasets in TensorFlow format.

## Remarks
- To use the Tensorboard visualization, you have to run "tensorboard --logdir 'path to tensorboard folder'"
- The binary files to get the benchmark of Tlite models can be found in this [link](https://www.tensorflow.org/lite/performance/measurement)
- For visualizing the tflite model, run : "python -m visualize 'model'.tflite visualized_model.html". It will generate an html file. This can be useful to see which operators are used in the tflite model.
