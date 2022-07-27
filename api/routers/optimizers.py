import time
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.celery_utils import get_task_info
from optimizers.pruning import Pruning
from tasks.tasks import apply_quantization

router = APIRouter(
    tags=['Optimizers'], responses={404: {"description": "Not found"}}
)


class OptimizeRequest(BaseModel):
    project_name: str
    project_path: str
    baseline_accuracy: float
    techniques: List[str]
    epoch: int
    batch_size: int
    learning_rate: float
    optimizer: str
    color_scheme: str


@router.post("/optimize")
async def optimize(req: OptimizeRequest):
    # Get current time in milliseconds
    initiated_time = int(time.time() * 1000)

    tasks = []

    for technique in req.techniques:
        if technique == "quantization":
            task = apply_quantization.apply_async(
                args=[
                    req.project_name, initiated_time, req.project_path,
                    req.baseline_accuracy, req.epoch, req.batch_size,
                    req.learning_rate, req.optimizer, req.color_scheme
                ]
            )
            tasks.append(
                {
                    "task_id": task.id,
                    "initiated_time": initiated_time,
                    "technique": "quantization",
                    "status": 'PENDING'
                }
            )
        if technique == 'prunning':
            pruned_model = Pruning(
                project_path=req.project_path, baseline_accuracy=req.baseline_accuracy,
                epoch=req.epoch, batch_size=req.batch_size, learning_rate=req.learning_rate, optimizer=req.optimizer
            )
            print('Pruning model created')

            pruned_model.compile_run()
            metrics = pruned_model.get_metrics()

            result = {
                "project_name": req.project_name,
                "project_path": req.project_path,
                "baseline_accuracy": req.baseline_accuracy,
                "epoch": req.epoch,
                "batch_size": req.batch_size,
                "learning_rate": req.learning_rate,
                "optimizer": req.optimizer,
                "technique": "Pruning",
                "initiated_time": initiated_time,
                "metrics": metrics
            }
            print(result)
        if technique == 'distillation':
            pass

    return JSONResponse(tasks)


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """
    Return the status of the submitted Task
    """
    return get_task_info(task_id)
