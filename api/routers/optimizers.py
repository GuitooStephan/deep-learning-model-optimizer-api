import time
from typing import List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session


from config.celery_utils import get_task_info
from tasks.tasks import apply_quantization, apply_pruning, apply_distillation
from db import crud
from tasks import tasks
from config.celery_utils import get_task_info, get_task_status as get_celery_task_status
from tasks.tasks import apply_quantization, apply_pruning
from db.database import SessionLocal
from db import crud

router = APIRouter(
    tags=['Optimizers'], responses={404: {"description": "Not found"}}
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class OptimizeRequest(BaseModel):
    project_id: str
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
async def optimize(req: OptimizeRequest, db: Session = Depends(get_db)):
    # Get current time in milliseconds
    initiated_time = int(time.time() * 1000)

    project_tasks = []
    for technique in req.techniques:
        if technique == "quantization":
            task = apply_quantization.apply_async(
                args=[
                    req.project_id, req.project_name, initiated_time, req.project_path,
                    req.baseline_accuracy, req.epoch, req.batch_size,
                    req.learning_rate, req.optimizer, req.color_scheme
                ]
            )
        if technique == 'pruning':
            task = apply_pruning.apply_async(
                args=[
                    req.project_id, req.project_name, initiated_time, req.project_path,
                    req.baseline_accuracy, req.epoch, req.batch_size,
                    req.learning_rate, req.optimizer, req.color_scheme
                ]
            )
        if technique == 'distillation':
            task = apply_distillation.apply_async(
                args=[
                    req.project_id, req.project_name, initiated_time, req.project_path,
                    req.baseline_accuracy, req.epoch, req.batch_size,
                    req.learning_rate, req.optimizer, req.color_scheme
                ]
            )
            tasks.append(
                {
                    "task_id": task.id,
                    "initiated_time": initiated_time,
                    "technique": "distillation",
                    "status": 'PENDING'
                }
            )

        project_tasks.append([req.project_id, req.project_name, task.id])

    crud.create_project_tasks(db, project_tasks)

    return JSONResponse({"project_id": req.project_id, "project_name": req.project_name, "project_status": "PENDING"})


class ProjectsStatus(BaseModel):
    projects_ids: List[str]


@router.post("/status/projects")
async def get_task_status(req: ProjectsStatus, db: Session = Depends(get_db)) -> dict:
    """
    Return the status of all the tasks for a given project
    """
    results = []

    for project_id in req.projects_ids:
        tasks = crud.get_project_tasks(db, project_id)
        tasks_status = [get_celery_task_status(task.task_id) for task in tasks]
        results.append({"project_id": project_id,
                        "project_name": tasks[0].project_name if len(tasks) else None,
                       "status": "DONE" if all(tasks_status) else "PENDING"})

    return JSONResponse(results)


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """
    Return the status of the submitted Task
    """
    return get_task_info(task_id)
