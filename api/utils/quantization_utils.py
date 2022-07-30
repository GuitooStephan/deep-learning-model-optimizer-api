from tensorflow.keras.layers import Dropout, BatchNormalization, Dense
import tensorflow_model_optimization as tfmot


def annotate_layer(layer):
    '''
        Annotate a layer with quantization information.
    '''
    if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization) or isinstance(layer, Dense):
        return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
