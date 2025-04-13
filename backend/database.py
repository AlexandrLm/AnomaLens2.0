import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./anomalens.db") # Значение по умолчанию на случай отсутствия .env

# Создаем движок SQLAlchemy
# connect_args необходим только для SQLite. В PostgreSQL он не нужен.
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Создаем SessionLocal класс для сессий базы данных
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Создаем базовый класс для моделей SQLAlchemy
Base = declarative_base()

# Функция для получения сессии БД (Dependency для FastAPI)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 