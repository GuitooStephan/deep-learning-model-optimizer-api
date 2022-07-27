from database import Base
from sqlalchemy.sql import func
from sqlalchemy import (String, Column)


class Project(Base):
    __tablename__ = 'project_info'
    projectId = Column(String(255))
    projectName = Column(String(255))
    taskId = Column(String(255))


    def __repr__(self):
        return f"<Project Id={self.projectId} Project name={self.projectName} taskId={self.taskId}>"