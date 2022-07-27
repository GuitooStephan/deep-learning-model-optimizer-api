from db.database import Base
from sqlalchemy.sql import func
from sqlalchemy import (String, Column)


class Project(Base):
    __tablename__ = 'project_info'
    taskId = Column(String(255), primary_key=True, index=True)
    projectId = Column(String(255))
    projectName = Column(String(255))
    
    
    def __repr__(self):
        return f"<taskId={self.taskId} Project Id={self.projectId} Project name={self.projectName} >"