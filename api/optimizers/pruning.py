import time
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
import numpy as np

from optimizers.optimizer import Optimizer


class Pruning(Optimizer):
    def __init__(self, project_path, baseline_accuracy, epoch, batch_size, learning_rate, optimizer, color_scheme):
        super().__init__(
            project_path, baseline_accuracy,
            epoch, batch_size, learning_rate, optimizer, color_scheme, 'pruning'
        )
        self.create_model()
        print('Pruning initialized')

    def create_model(self):
        self.model = tf.keras.models.clone_model(
            self.baseline_model,
        )

        num_images = self.X_train.shape[0]
        end_step = np.ceil(
            num_images / self.batch_size).astype(np.int32) * self.epoch
        print(end_step)

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.80,
                begin_step=0,
                end_step=end_step
            )
        }

        self.model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model, **pruning_params
        )

    def compile_run(self):
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=self.optimizer
        )

        training_st = time.process_time()
        self.hist = self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=2,
            validation_data=(self.X_val, self.y_val),
            callbacks=tfmot.sparsity.keras.UpdatePruningStep()
        )
        training_et = time.process_time()
        self.training_time = training_et - training_st

        self.strip_model_export()
        self.save_model()

    def get_metrics(self):
        super().get_metrics()
        self.metrics['pruned_model_size'] = self.get_model_size(
            os.path.join(self.project_path, 'pruned_model.h5')
        )
        self.get_number_of_parameters()
        return self.metrics

    def strip_model_export(self):
        self.pruned_model = tfmot.sparsity.keras.strip_pruning(self.model)
        # self.pruned_model.compile(
        #     loss=tf.keras.losses.SparseCategoricalCrossentropy(
        #         from_logits=True),
        #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        #     optimizer=self.optimizer
        # )

    def get_number_of_parameters(self):
        # Remove the weights not equal to 0
        total = 0
        zeros = 0
        for i, w in enumerate(self.pruned_model.get_weights()):
            total = total + w.size
            zeros = zeros + np.sum(w == 0)
        self.metrics['parameters'] = int(total - zeros)

    def save_model(self):
        self.pruned_model.save(os.path.join(
            self.project_path, 'pruned_model.h5'))
