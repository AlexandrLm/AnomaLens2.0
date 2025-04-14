from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func  # Для значения по умолчанию CURRENT_TIMESTAMP
import json # Для работы с JSON в details
from sqlalchemy.types import TypeDecorator

from .database import Base # Импортируем Base из нашего database.py

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)

    products = relationship("Product", back_populates="category")

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=True)
    price = Column(Float, nullable=False)
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True) # Может быть товар без категории? Или nullable=False? Пока True.

    category = relationship("Category", back_populates="products")
    order_items = relationship("OrderItem", back_populates="product")

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    # Добавим поля для возможного анализа
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # Можно добавить больше полей: address, phone, etc.

    orders = relationship("Order", back_populates="customer")
    activities = relationship("UserActivity", back_populates="customer")

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    total_amount = Column(Float, nullable=False) # Рассчитывается при создании заказа
    status = Column(String, default="pending") # Например: pending, processing, shipped, delivered, cancelled

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False) # Цена товара на момент заказа

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")

class UserActivity(Base):
    __tablename__ = "user_activities"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True, index=True) # Может быть null для анонимных пользователей
    session_id = Column(String, index=True, nullable=True) # Для отслеживания сессий
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    action_type = Column(String, nullable=False, index=True) # Например: view_product, add_to_cart, login, purchase, search
    details = Column(JSON, nullable=True) # Дополнительная информация (product_id, search_query, page_url, ip_address, user_agent, etc.)
    # Возможно, стоит добавить IP адрес и User Agent отдельными полями для индексации?
    ip_address = Column(String, nullable=True, index=True)
    user_agent = Column(String, nullable=True)

    customer = relationship("Customer", back_populates="activities")

class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(String, index=True, nullable=False) # Например, 'order', 'activity'
    entity_id = Column(Integer, index=True, nullable=False) # ID заказа, активности и т.д.
    detection_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    description = Column(Text, nullable=False) # Описание аномалии
    severity = Column(String, nullable=True, index=True) # Например, 'low', 'medium', 'high'
    detector_name = Column(String, nullable=True) # Какой алгоритм обнаружил
    details = Column(JSON, nullable=True) # Дополнительные данные об аномалии (например, значение метрики)
    anomaly_score = Column(Float, nullable=True) # Для хранения численной оценки аномальности

    # Можно добавить ForeignKey связи, если нужно жестко связать с entity_id
    # Например, order_id = Column(Integer, ForeignKey("orders.id"), nullable=True)
    # Но для гибкости оставим пока entity_type/entity_id 

# Тип для хранения JSON в текстовом поле (для SQLite/PostgreSQL)
class JsonEncodedDict(TypeDecorator):
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                 value = {} # Возвращаем пустой dict при ошибке декодирования
        return value

# Новая модель для настроек
class Setting(Base):
    __tablename__ = "settings"
    # Ключ настройки (уникальное имя)
    key = Column(String, primary_key=True, index=True)
    # Значение настройки (храним как строку, конвертируем при использовании)
    value = Column(String)

    def __repr__(self):
        return f"<Setting(key='{self.key}', value='{self.value}')>" 