from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense
import tensorflow_model_optimization as tfmot
import tensorflow_model_optimization as tfmot

from Optimizer import Optimizer


class Quantization(Optimizer):

    def __init__(self,epochs,batch_size, learning_rate,input_shape):
        super().__init__(epochs,batch_size, learning_rate,input_shape)
        self.create_blueprint()

    def create_blueprint(self):
        
        annotate = tfmot.quantization.keras.quantize_annotate_layer

        cnn_quantized = Sequential([
            annotate(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=self.input_shape)),
            annotate(Activation('relu')),
            annotate(Conv2D(32, kernel_size=(3, 3),padding='same')),
            annotate(Activation('relu')),
            BatchNormalization(),
            annotate(MaxPooling2D(pool_size=(2, 2),strides=2)),
            Dropout(0.2),
            annotate(Conv2D(64, kernel_size=(3, 3))),
            annotate(Activation('relu')),
            annotate(Conv2D(64, kernel_size=(3, 3))),
            annotate(Activation('relu')),
            BatchNormalization(),
            annotate(MaxPooling2D(pool_size=(2, 2),strides=2)),
            Dropout(0.3),
            Flatten(),
            annotate(Dense(256)),
            annotate(Activation('relu')),
            BatchNormalization(),
            Dropout(0.7),
            annotate(Dense(10)),
            Activation('softmax'),
        ])

        self.blueprint = tfmot.quantization.keras.quantize_apply(cnn_quantized)



