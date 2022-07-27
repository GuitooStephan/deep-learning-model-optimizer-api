from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
import numpy as np

from Optimizer import Optimizer


class Pruning(Optimizer):

    def __init__(self, project_path, baseline_accuracy, epoch, batch_size, learning_rate, optimizer):
        super().__init__(project_path, baseline_accuracy,
                         epoch, batch_size, learning_rate, optimizer)
        self.create_model()
        print('Pruning initialized')

    def create_model(self):

        pruning_model = tf.keras.models.clone_model(
            self.baseline_model,
            # clone_function=annotate_layer
        )

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.75,
                begin_step=0,
                end_step=15000
            )}

        self.model = tfmot.sparsity.keras.prune_low_magnitude(
            pruning_model, **pruning_params
        )

    def compile_run(self):
        super().compile_run()

        self.save_model()

    def get_metrics(self):
        super().get_metrics()
        self.metrics['pruned_model_size'] = self.get_model_size(
            os.path.join(self.project_path, 'pruned_model.h5')
        )
        return self.metrics

    def strip_model_export(self):

        pruned_model_for_export = tfmot.sparsity.keras.strip_pruning(
            self.model)
        pruned_model_for_export.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=Adam(lr=10**-3)
        )
        pruned_model_for_export.save(os.path.join(
            self.project_path, 'cnn_pruning.h5'))

    def get_params(self):

        pruned_model_for_export = tf.keras.models.load_model(
            os.path.join(self.project_path, 'cnn_pruning.h5'))

        # Remove the weights not equal to 0
        total = 0
        zeros = 0
        for i, w in enumerate(pruned_model_for_export.get_weights()):
            total = total + w.size
            zeros = zeros + np.sum(w == 0)
        self.params = total - zeros
        return self.params

    def save_model(self):
        self.model.save(os.path.join(self.project_path, 'pruned_model.h5'))
