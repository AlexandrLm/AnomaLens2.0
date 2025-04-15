from fastapi import FastAPI
from .database import engine, Base
from . import models # Импортируем модели, чтобы SQLAlchemy знало о них
from .api import store # Импортируем наш новый роутер
from .api import activities # Импортируем роутер активностей
from .api import simulator # Импортируем роутер симулятора
from .api import anomalies # Импортируем роутер аномалий
from .api import dashboard # <--- Добавляем импорт нового роутера
from .api import charts # <-- Импортируем роутер для графиков
from .api import orders # <--- Импортируем новый роутер

# Создаем таблицы в базе данных
# В реальном приложении лучше использовать Alembic для миграций
# Но для старта этого достаточно
try:
    print("Attempting to create database tables...")
    models.Base.metadata.create_all(bind=engine)
    print("Database tables checked/created successfully.")
except Exception as e:
    print(f"Error creating database tables: {e}")

# Создаем экземпляр FastAPI
app = FastAPI(
    title="AnomaLens API",
    description="API для платформы обнаружения аномалий AnomaLens",
    version="0.1.0",
)

# Базовый эндпоинт для проверки
@app.get("/")
async def read_root():
    return {"message": "Добро пожаловать в AnomaLens API!"}

# Подключаем роутеры из папки api/
app.include_router(store.router, prefix="/api/store", tags=["Store - Categories, Products, Customers, Orders"])
app.include_router(activities.router, prefix="/api/activities", tags=["User Activities"])
app.include_router(simulator.router, prefix="/api/simulator", tags=["Data Simulator"])
app.include_router(anomalies.router, prefix="/api/anomalies", tags=["Anomaly Detection"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"]) # <--- Добавляем новый роутер
app.include_router(charts.router, prefix="/api/charts", tags=["Charts"]) # <-- Добавляем роутер для графиков
app.include_router(orders.router, prefix="/api/order_data", tags=["Orders"]) # <--- Подключаем роутер заказов

# Другие роутеры (например, для аномалий) будут добавляться здесь
# app.include_router(anomalies_router.router, prefix="/api/anomalies", tags=["Anomalies"])

# --- CORS (оставляем включенным) ---
from fastapi.middleware.cors import CORSMiddleware
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------

# Инструкция для запуска:
# В терминале, находясь в папке backend и с активным .venv: .venv\Scripts\activate
# uvicorn backend.main:app --reload