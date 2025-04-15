# #############################################################################
# quantization.py
# =================
# Author :
# Jonathan REYMOND [jonathan.reymond7@gmail.com]
# #############################################################################

import tensorflow_model_optimization as tfmot

from tensorflow_model_optimization.quantization.keras.quantizers import MovingAverageQuantizer, LastValueQuantizer
from tensorflow_model_optimization.quantization.keras import QuantizeConfig




class InversionLayerQuantizeConfig(QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):

        weights_and_quantizers = [(layer_weights, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))
                                  for layer_weights in layer.get_list_weights()]
        
        return weights_and_quantizers
                
    

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order
        layer.set_list_weights(quantize_weights)

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        layer.activation = quantize_activations[0]


    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        # Does not quantize output, since we return an empty list.
        return []

    def get_config(self):
      return {}