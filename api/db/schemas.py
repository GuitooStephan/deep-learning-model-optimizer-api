from pydantic import BaseModel


class Project(BaseModel):
    projectId: str
    projectName: str
    taskId: str
 
    class Config:
        orm_mode = True