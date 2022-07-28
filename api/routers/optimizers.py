import time
from typing import List
from fastapi import APIRouter,Depends
from pydantic import BaseModel
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session


from config.celery_utils import get_task_info
from tasks.tasks import apply_quantization, apply_pruning, apply_distillation
from db.database import SessionLocal, engine
from db import db_models
from db import crud
from db import database
from tasks import tasks

router = APIRouter(
    tags=['Optimizers'], responses={404: {"description": "Not found"}}
)

db = database.SessionLocal()

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
async def optimize(req: OptimizeRequest):
    # Get current time in milliseconds
    initiated_time = int(time.time() * 1000)

    tasks = []
    projectList=[]
    for technique in req.techniques:
        if technique == "quantization":
            task = apply_quantization.apply_async(
                args=[
                    req.project_id,req.project_name, initiated_time, req.project_path,
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
        if technique == 'pruning':
            task = apply_pruning.apply_async(
                args=[
                    req.project_id,req.project_name, initiated_time, req.project_path,
                    req.baseline_accuracy, req.epoch, req.batch_size,
                    req.learning_rate, req.optimizer, req.color_scheme
                ]
            )
            tasks.append(
                {
                    "task_id": task.id,
                    "initiated_time": initiated_time,
                    "technique": "pruning",
                    "status": 'PENDING'
                }
            )
        if technique == 'distillation':
            task = apply_distillation.apply_async(
                args=[
                    req.project_id,req.project_name, initiated_time, req.project_path,
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

        projectList.append([req.project_id,req.project_name, task.id])
    
    crud.create_project_task_IDs(db,projectList)

    return JSONResponse({"project_id":req.project_id,"project_name":req.project_name,"project_status":"PENDING"})


@router.get("/task/{task_id}")
async def get_task_status(task_id: str) -> dict:
    """
    Return the status of the submitted Task
    """
    return get_task_info(task_id)

@router.post("/optimize_test")
async def optimize_test(req: OptimizeRequest):
    tasks.apply_distillation(req.project_id,req.project_name, req.project_path,
                    req.baseline_accuracy, req.epoch, req.batch_size,
                    req.learning_rate, req.optimizer, req.color_scheme,req.techniques[0])
