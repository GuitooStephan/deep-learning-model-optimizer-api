from db.database import Base
from sqlalchemy.sql import func
from sqlalchemy import (String, Column, Integer, DateTime)


class ProjectTasks(Base):
    __tablename__ = 'project_info'
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255))
    project_id = Column(String(255))
    project_name = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=func.now())

    def __repr__(self):
        return f"<taskId={self.task_id} Project Id={self.project_id} Project name={self.project_name} >"
