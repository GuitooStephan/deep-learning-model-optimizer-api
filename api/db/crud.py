from sqlalchemy import desc
from sqlalchemy.orm import Session
from db import db_models


def create_project_task_IDs(db: Session,Projectlist):

    tasklist=[]

    for i in range(0, len(Projectlist)):
        projectId, projectName, taskId = Projectlist[i]
        tasklist.append(db_models.Project(
                taskId = taskId,
                projectId = projectId,
                projectName = projectName
            ))

    db.bulk_save_objects(tasklist)
    db.commit()