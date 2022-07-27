from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

engine=create_engine("postgresql://postgres:postgres@localhost/project_tasks",
    echo=True
)

print("database connection made")


if not database_exists(engine.url):
    create_database(engine.url)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


