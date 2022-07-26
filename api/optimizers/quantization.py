import os
import numpy as np

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import lite

from utils.quantization_utils import annotate_layer
from optimizers.optimizer import Optimizer


class Quantization(Optimizer):
    def __init__(self, project_path, baseline_accuracy, epoch, batch_size, learning_rate, optimizer, color_scheme):
        super().__init__(
            project_path, baseline_accuracy,
            epoch, batch_size, learning_rate,
            optimizer, color_scheme)
        self.create_model()
        print('Quantization initialized')

    def create_model(self):
        self.model = tf.keras.models.clone_model(
            self.baseline_model,
            clone_function=annotate_layer
        )

        self.model = tfmot.quantization.keras.quantize_apply(self.model)

    def compile_run(self):
        super().compile_run()
        # Apply quantization to the model
        converter = lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [lite.Optimize.DEFAULT]
        self.model_lite = converter.convert()
        self.save_model()

    def get_metrics(self):
        super().get_metrics()
        self.get_accuracy_of_lite_model()
        self.metrics['quantized_model_size'] = self.get_model_size(
            os.path.join(self.project_path, 'quantized_model.h5')
        )
        self.metrics['test accuracy'] = self.lite_model_accuracy
        return self.metrics

    def get_accuracy_of_lite_model(self):
        interpreter = lite.Interpreter(model_content=self.model_lite)
        interpreter.allocate_tensors()
        # Assigning Input & Output Index Details to Tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        acc = 0
        for i in range(len(self.X_test)):
            input_data = self.X_test[i].reshape(input_shape)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            y_pred = np.argmax(output_data)
            y_true = self.y_test[i]
            if(y_pred == y_true):
                acc += 1
        self.lite_model_accuracy = acc/len(self.X_test)

    def save_model(self):

        self.model.save(os.path.join(self.project_path, 'quantized_model.h5'))
        open(os.path.join(self.project_path, "quantized_model_lite.tflite"),
             "wb").write(self.model_lite)
