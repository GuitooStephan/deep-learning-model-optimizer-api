
from celery import shared_task

from optimizers.quantization import Quantization


# Initiate quantization
@shared_task(
    bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2},
    name='quantization:apply_quantization'
)
def apply_quantization(
        self, project_name: str, initiated_time: int, project_path: str,
        baseline_accuracy: float, epoch: int, batch_size: int, learning_rate: float, optimizer: str,):

    print('Task quantization:apply_quantization')
    # Creating a object of the chosen optimizer
    quantized_model = Quantization(
        project_path=project_path, baseline_accuracy=baseline_accuracy,
        epoch=epoch, batch_size=batch_size, learning_rate=learning_rate, optimizer=optimizer
    )
    print('Quantization model created')
    quantized_model.compile_run()
    metrics = quantized_model.get_metrics()

    result = {
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
