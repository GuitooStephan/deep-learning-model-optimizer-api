from sqlalchemy import desc
from sqlalchemy.orm import Session
from db import db_models


def create_project_tasks(db: Session, project_tasks):

    tasklist = []

    for i in range(0, len(project_tasks)):
        project_id, project_name, task_id = project_tasks[i]
        tasklist.append(db_models.ProjectTasks(
            task_id=task_id,
            project_id=project_id,
            project_name=project_name
        ))

    db.bulk_save_objects(tasklist)
    db.commit()


def get_project_tasks(db: Session, project_id: str):
    return db.query(
        db_models.ProjectTasks
    ).where(
        db_models.ProjectTasks.project_id == project_id
    ).all()
