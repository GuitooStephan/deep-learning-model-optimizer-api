
from numpy import str_
from celery import shared_task

from optimizers.quantization import Quantization
from optimizers.pruning import Pruning
from optimizers.distillation import Distillation, Distiller

import tensorflow_model_optimization as tfmot


# Initiate quantization
@shared_task(
    bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
    name='quantization:apply_quantization'
)
def apply_quantization(
        self, project_id: str, project_name: str, initiated_time: int, project_path: str,
        baseline_accuracy: float, epoch: int, batch_size: int,
        learning_rate: float, optimizer: str, color_scheme: str):

    # Creating a object of the chosen optimizer
    quantized_model = Quantization(
        project_path, baseline_accuracy, epoch,
        batch_size, learning_rate, optimizer,
        color_scheme
    )
    quantized_model.compile_run()
    metrics = quantized_model.get_metrics()

    result = {
        "project_id": project_id,
        "project_name": project_name,
        "project_path": project_path,
        "baseline_accuracy": baseline_accuracy,
        "epoch": epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "technique": "quantization",
        "initiated_time": initiated_time,
        "metrics": metrics
    }
    return result

# Initiate Pruning


@shared_task(
    bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
    name='pruning:apply_pruning'
)
def apply_pruning(
        self, project_id: str, project_name: str, initiated_time: int, project_path: str,
        baseline_accuracy: float, epoch: int, batch_size: int,
        learning_rate: float, optimizer: str, color_scheme: str):

    # Creating a object of the chosen optimizer
    pruned_model = Pruning(
        project_path=project_path, baseline_accuracy=baseline_accuracy,
        epoch=epoch, batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer,
        color_scheme=color_scheme
    )

    pruned_model.compile_run()
    metrics = pruned_model.get_metrics()

    result = {
        "project_id": project_id,
        "project_name": project_name,
        "project_path": project_path,
        "baseline_accuracy": baseline_accuracy,
        "epoch": epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "technique": "pruning",
        "initiated_time": initiated_time,
        "metrics": metrics
    }

    return result


# Initiate Distillation
@shared_task(
    bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
    name='distillation:apply_distillation'
)
def apply_distillation(
        self, project_id: str, project_name: str, initiated_time: int, project_path: str,
        baseline_accuracy: float, epoch: int, batch_size: int,
        learning_rate: float, optimizer: str, color_scheme: str):

    print('Task distillation:apply_distillation')

    # Creating a object of the chosen optimizer
    distilled_model = Distillation(
        project_path=project_path, baseline_accuracy=baseline_accuracy,
        epoch=epoch, batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer, color_scheme=color_scheme
    )
    print('Teacher model created')

    metrics = distilled_model.get_metrics()

    result = {
        "project_id": project_id,
        "project_name": project_name,
        "project_path": project_path,
        "baseline_accuracy": baseline_accuracy,
        "epoch": epoch,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "technique": "distillation",
        "initiated_time": initiated_time,
        "metrics": metrics
    }

    return result
