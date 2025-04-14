from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime # Нужно для created_at

# ==================
# Базовые Сущности (переносим Product и Category ниже для ясности зависимостей)
# ==================

# Заготовка для OrderItem (нужна для Product)
class OrderItem(BaseModel):
    id: int
    order_id: int
    product_id: int
    quantity: int
    price: float
    model_config = ConfigDict(from_attributes=True)

# Заготовка для Order (нужна для Customer)
class Order(BaseModel):
    id: int
    customer_id: int
    created_at: datetime
    total_amount: float
    status: str
    items: List[OrderItem] = []
    model_config = ConfigDict(from_attributes=True)

# Заготовка для UserActivity (нужна для Customer)
class UserActivity(BaseModel):
    id: int
    # ... другие поля UserActivity для чтения
    model_config = ConfigDict(from_attributes=True)

# ==================
# Category
# ==================
class CategoryBase(BaseModel):
    name: str
    description: Optional[str] = None

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase): # Схема для чтения
    id: int
    # products: List['Product'] = [] # Пока не добавляем для простоты
    model_config = ConfigDict(from_attributes=True)

# ==================
# Product
# ==================
class ProductBase(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    category_id: Optional[int] = None

class ProductCreate(ProductBase):
    pass

class Product(ProductBase): # Схема для чтения
    id: int
    category: Optional[Category] = None # Отображаем связанную категорию
    # order_items: List[OrderItem] = [] # Можно добавить, но пока не будем усложнять
    model_config = ConfigDict(from_attributes=True)

# ==================
# Customer
# ==================
class CustomerBase(BaseModel):
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class CustomerCreate(CustomerBase):
    pass

class Customer(CustomerBase): # Схема для чтения
    id: int
    created_at: datetime
    orders: List[Order] = [] # Отображаем список заказов клиента
    # activities: List[UserActivity] = [] # Можно добавить позже
    model_config = ConfigDict(from_attributes=True)

# ==================
# OrderItem
# ==================
# Используется при создании заказа
class OrderItemCreate(BaseModel):
    product_id: int
    quantity: int
    # Цена не указывается при создании, она берется из товара на момент создания заказа

# Переопределяем OrderItem с полным набором полей и связью на Product
class OrderItem(BaseModel): # Схема для чтения
    id: int
    order_id: int
    product_id: int
    quantity: int
    price: float # Цена на момент заказа
    product: Product # Отображаем информацию о товаре
    model_config = ConfigDict(from_attributes=True)

# ==================
# Order
# ==================
# Используется при создании заказа
class OrderCreate(BaseModel):
    customer_id: int
    items: List[OrderItemCreate] # Список товаров и их количество

# Переопределяем Order с полным набором полей и связями Customer/OrderItem
class Order(BaseModel): # Схема для чтения
    id: int
    customer_id: int
    created_at: datetime
    total_amount: float
    status: str
    customer: Customer # Отображаем информацию о клиенте
    items: List[OrderItem] = [] # Отображаем список позиций заказа
    model_config = ConfigDict(from_attributes=True)

# ==================
# UserActivity (заготовка)
# ==================
class UserActivityBase(BaseModel):
    customer_id: Optional[int] = None
    session_id: Optional[str] = None
    action_type: str
    details: Optional[dict] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class UserActivityCreate(UserActivityBase):
    pass

# Переопределяем UserActivity для чтения
class UserActivity(UserActivityBase):
    id: int
    timestamp: datetime
    customer: Optional[Customer] = None # Связь с клиентом (если есть)
    model_config = ConfigDict(from_attributes=True)

# ==================
# Anomaly
# ==================
class AnomalyBase(BaseModel):
    entity_type: str # 'order', 'activity', etc.
    entity_id: int   # ID of the corresponding order or activity
    description: str # Renamed from reason to description for clarity in DB
    severity: Optional[str] = None
    detector_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    anomaly_score: Optional[float] = None # Added anomaly_score here too

class AnomalyCreate(AnomalyBase):
    # Inherits fields from AnomalyBase
    pass

class Anomaly(AnomalyBase): # Schema for reading single Anomaly records
    id: int
    detection_timestamp: datetime
    # description field is inherited
    model_config = ConfigDict(from_attributes=True)

# Обновляем зависимые схемы, если Pydantic не справился автоматически (обычно v2 справляется)
# Customer.model_rebuild()
# Product.model_rebuild()

# --- Новая схема для детального ответа по аномалии ---
class AnomalyDetailResponse(BaseModel):
    anomaly: Anomaly # Данные самой аномалии
    related_entity: Optional[Union[UserActivity, Order]] = None # Данные связанной сущности

# --- Схемы для Параметров и Ответов API --- 

# ВОЗВРАЩАЕМ ПАРАМЕТРЫ НАСТРОЕК
class TrainingParams(BaseModel):
   limit: int = Field(10000, ge=1, description="Maximum number of records for training.")
   z_threshold: float = Field(3.0, gt=0, description="Z-score threshold for statistical detector.")
   # Оставляем entity_type необязательным, чтобы кнопка "обучить все" работала
   entity_type: Optional[str] = Field(None, description="Entity type to train ('activity', 'order'). If None, trains all.")

# ВОЗВРАЩАЕМ ПАРАМЕТРЫ НАСТРОЕК
class DetectionParams(BaseModel):
    entity_type: str = Field(..., description="The type of entity to detect anomalies for ('activity' or 'order').")
    algorithms: List[str] = Field(..., description="List of algorithm names to run (e.g., ['statistical_zscore', 'isolation_forest']).")
    limit: int = Field(10000, ge=1, description="Maximum number of recent records to analyze.")
    z_threshold: Optional[float] = Field(3.0, gt=0, description="Z-score threshold for statistical detector (if used).")
    dbscan_eps: Optional[float] = Field(0.5, gt=0, description="DBSCAN epsilon parameter (if used).")
    dbscan_min_samples: Optional[int] = Field(5, ge=1, description="DBSCAN min_samples parameter (if used).")

class DetectResponse(BaseModel):
    message: str
    anomalies_saved_by_algorithm: Dict[str, int]

# Модели для данных симулятора (оставляем как есть)
class SimulatorRequest(BaseModel):
    num_customers: int = Field(100, ge=1)
    num_products: int = Field(50, ge=1)
    num_activities: int = Field(1000, ge=1)
    num_orders: int = Field(200, ge=1)
    anomaly_rate: float = Field(0.02, ge=0, le=1)
    # Параметры для аномалий в заказах
    high_order_amount_threshold: float = Field(500.0, ge=0)
    high_item_count_threshold: int = Field(10, ge=1)
    order_anomaly_rate: float = Field(0.03, ge=0, le=1)

class SimulatorResponse(BaseModel):
    message: str
    customers_added: int
    products_added: int
    activities_added: int
    orders_added: int
    anomalous_activities_inserted: int
    anomalous_orders_inserted: int

# --- Схема для информации от одного детектора (используется в ConsolidatedAnomaly) --- 
class AnomalyDetectorInfo(BaseModel):
    detector_name: str
    anomaly_score: Optional[float] = None
    severity: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    detection_timestamp: datetime 
    # --- ДОБАВЛЕНО ПОЛЕ reason --- 
    reason: Optional[str] = None # Это поле теперь будет включаться в ответ API
    # ----------------------------

    model_config = ConfigDict(from_attributes=True)

# --- Схема для консолидированной аномалии --- 
class ConsolidatedAnomaly(BaseModel):
    entity_type: str
    entity_id: int
    last_detected_at: datetime
    overall_severity: Optional[str] = None
    detector_count: int
    triggered_detectors: List[AnomalyDetectorInfo] # Эта схема теперь содержит reason

# --- Обновляем схему ответа для эндпоинта GET / --- 
# Вместо List[Anomaly] или AnomalyResponse, будем использовать List[ConsolidatedAnomaly]

# --- Схемы для Деталей Аномалии (GET /{anomaly_id}) --- 
# Их можно пока оставить без изменений, т.к. они запрашивают конкретную запись по ID
class AnomalyDetailResponse(BaseModel):
    anomaly: Anomaly 
    related_entity: Optional[Union[UserActivity, Order]] = None