import os
from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
# from sqlalchemy_utils import drop_database


load_dotenv()
engine = create_engine(
    os.environ.get(
        'POSTGRES_DATABASE_URL',
        'postgresql://postgres:postgres@localhost/Project_metricInfo'
    ),
    echo=True
)

# Delete the whole database if exists
# Uncomment if needed
# if database_exists(engine.url):
# drop_database(engine.url)

if not database_exists(engine.url):
    create_database(engine.url)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()