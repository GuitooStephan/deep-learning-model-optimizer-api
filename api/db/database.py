import os
from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

load_dotenv()

engine = create_engine(
    os.environ.get(
        "POSTGRES_DATABASE_URL",
        "postgresql://postgres:postgres@localhost/project_tasks"
    ),
    echo=True
)

print("database connection made")


if not database_exists(engine.url):
    create_database(engine.url)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()
