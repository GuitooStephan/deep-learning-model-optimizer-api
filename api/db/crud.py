from sqlalchemy import desc
from sqlalchemy.orm import Session
import db_models


def create_project_task_IDs(db: Session,tasklist):

    projectId, projectName, taskId = tasklist
    new_project = db_models.Project(
        projectId = projectId,
        projectName = projectName,
        taskId = taskId
    )

    db.add(new_project)
    db.commit()