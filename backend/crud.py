from sqlalchemy.orm import Session, joinedload
from sqlalchemy.orm import subqueryload
from decimal import Decimal
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from sqlalchemy import func, cast, Date

from . import models, schemas

# === Category CRUD ===

def get_category(db: Session, category_id: int):
    """Получает категорию по ID."""
    return db.query(models.Category).filter(models.Category.id == category_id).first()

def get_category_by_name(db: Session, name: str):
    """Получает категорию по имени (для проверки уникальности)."""
    return db.query(models.Category).filter(models.Category.name == name).first()

def get_categories(db: Session, skip: int = 0, limit: int = 100):
    """Получает список категорий с пагинацией."""
    return db.query(models.Category).offset(skip).limit(limit).all()

def create_category(db: Session, category: schemas.CategoryCreate):
    """Создает новую категорию."""
    db_category = models.Category(name=category.name, description=category.description)
    db.add(db_category)
    db.commit()
    db.refresh(db_category)
    return db_category

# Функции для Update и Delete добавим позже, если понадобятся

# === Product CRUD ===

def get_product(db: Session, product_id: int):
    """Получает продукт по ID."""
    return db.query(models.Product).filter(models.Product.id == product_id).first()

def get_products(db: Session, skip: int = 0, limit: int = 100):
    """Получает список продуктов с пагинацией."""
    # Используем joinedload для загрузки связанной категории сразу, чтобы избежать N+1 запросов
    return db.query(models.Product).options(joinedload(models.Product.category)).offset(skip).limit(limit).all()

def create_product(db: Session, product: schemas.ProductCreate):
    """Создает новый продукт."""
    # Проверяем, существует ли категория, если category_id указан
    if product.category_id:
        db_category = get_category(db, category_id=product.category_id)
        if not db_category:
            # Можно либо вернуть ошибку, либо создать продукт без категории
            # Пока вернем ошибку для ясности
            raise ValueError(f"Category with id {product.category_id} not found") # Эта ошибка будет обработана в API

    db_product = models.Product(
        name=product.name,
        description=product.description,
        price=product.price,
        category_id=product.category_id
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    # Загружаем категорию перед возвратом, чтобы она была в ответе
    db.refresh(db_product, attribute_names=['category'])
    return db_product

# Функции для Update и Delete Product добавим позже 

# === Customer CRUD ===

def get_customer(db: Session, customer_id: int):
    """Получает клиента по ID."""
    return db.query(models.Customer).filter(models.Customer.id == customer_id).first()

def get_customer_by_email(db: Session, email: str):
    """Получает клиента по email (для проверки уникальности)."""
    return db.query(models.Customer).filter(models.Customer.email == email).first()

def get_customers(db: Session, skip: int = 0, limit: int = 100):
    """Получает список клиентов с пагинацией."""
    return db.query(models.Customer).offset(skip).limit(limit).all()

def create_customer(db: Session, customer: schemas.CustomerCreate):
    """Создает нового клиента."""
    # В реальном приложении здесь была бы логика хеширования пароля
    db_customer = models.Customer(
        email=customer.email,
        first_name=customer.first_name,
        last_name=customer.last_name
        # created_at установится автоматически через server_default
    )
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    return db_customer

# Функции для Update и Delete Customer добавим позже

# === Order CRUD ===

def get_order(db: Session, order_id: int):
    """
    Получает заказ по ID.
    Загружает связанного клиента и позиции заказа (включая товары в позициях).
    """
    return (
        db.query(models.Order)
        .options(
            joinedload(models.Order.customer), # Загружаем клиента
            subqueryload(models.Order.items).joinedload(models.OrderItem.product) # Загружаем позиции и их товары
        )
        .filter(models.Order.id == order_id)
        .first()
    )

def get_orders(db: Session, skip: int = 0, limit: int = 100, customer_id: Optional[int] = None):
    """
    Получает список заказов с пагинацией.
    Опционально фильтрует по customer_id.
    Загружает связанного клиента и позиции заказа (включая товары).
    """
    query = db.query(models.Order).options(
        joinedload(models.Order.customer),
        subqueryload(models.Order.items).joinedload(models.OrderItem.product)
    )
    if customer_id is not None:
        query = query.filter(models.Order.customer_id == customer_id)
    return query.offset(skip).limit(limit).all()

def create_order(db: Session, order: schemas.OrderCreate, items_data: List[schemas.OrderItemCreate]):
    """
    Создает новый заказ и связанные с ним позиции заказа.
    Проверяет наличие клиента и товаров.
    Рассчитывает общую стоимость заказа.
    """
    # 1. Проверить существование клиента
    db_customer = get_customer(db, customer_id=order.customer_id)
    if not db_customer:
        raise ValueError(f"Customer with id {order.customer_id} not found")

    # 2. Обработать позиции заказа и рассчитать общую сумму
    total_amount = Decimal("0.0")
    order_items_data = [] # Список для хранения данных для создания OrderItem

    if not items_data:
        raise ValueError("Order must contain at least one item")

    for item_data in items_data:
        # Проверить существование товара
        db_product = get_product(db, product_id=item_data.product_id)
        if not db_product:
            raise ValueError(f"Product with id {item_data.product_id} not found")

        # Проверить количество (должно быть > 0)
        if item_data.quantity <= 0:
            raise ValueError(f"Quantity for product id {item_data.product_id} must be positive")

        # Получить актуальную цену товара
        current_price = Decimal(str(db_product.price)) # Преобразуем float в Decimal
        item_total = current_price * item_data.quantity
        total_amount += item_total

        order_items_data.append({
            "product_id": db_product.id,
            "quantity": item_data.quantity,
            "price": float(current_price), # Сохраняем как float в БД (согласно модели)
            "product": db_product # Передаем модель для связи
        })

    # 3. Создать объект заказа
    db_order = models.Order(
        customer_id=order.customer_id,
        total_amount=float(total_amount), # Сохраняем как float в БД
        status="pending", # Статус по умолчанию
        customer=db_customer # Связываем с моделью клиента
    )
    db.add(db_order)
    # Нужно "протолкнуть" изменения, чтобы получить order.id для OrderItem
    # Можно использовать flush или commit+refresh
    db.flush() # Получаем ID для db_order перед созданием items

    # 4. Создать объекты позиций заказа
    for item_to_create in order_items_data:
        db_order_item = models.OrderItem(
            order_id=db_order.id,
            product_id=item_to_create["product_id"],
            quantity=item_to_create["quantity"],
            price=item_to_create["price"],
            order=db_order, # Связываем с моделью заказа
            product=item_to_create["product"] # Связываем с моделью товара
        )
        db.add(db_order_item)

    # 5. Сохранить все изменения
    db.commit()

    # 6. Обновить объект заказа, чтобы загрузить связанные позиции
    db.refresh(db_order)
    # Загружаем остальные связи для полного ответа
    db.refresh(db_order, attribute_names=['items', 'customer'])
    for item in db_order.items:
         db.refresh(item, attribute_names=['product'])

    return db_order

# Функции для OrderItem обычно не нужны отдельно, т.к. они управляются через Order.
# Если понадобятся (например, get_order_item), можно добавить.

# === UserActivity CRUD ===

def create_user_activity(db: Session, activity: schemas.UserActivityCreate):
    """Создает новую запись об активности пользователя."""
    # Проверяем, существует ли customer_id, если он указан
    if activity.customer_id:
        db_customer = get_customer(db, customer_id=activity.customer_id)
        if not db_customer:
            # Решаем, что делать: ошибка или null? ТЗ допускает Null, оставим null
            activity.customer_id = None
            # Можно логировать предупреждение, что customer_id не найден

    db_activity = models.UserActivity(
        customer_id=activity.customer_id,
        session_id=activity.session_id,
        action_type=activity.action_type,
        details=activity.details,
        ip_address=activity.ip_address,
        user_agent=activity.user_agent
        # timestamp установится автоматически через server_default
    )
    db.add(db_activity)
    db.commit()
    db.refresh(db_activity)
    # Загружаем клиента, если он есть, для ответа
    if db_activity.customer_id:
        db.refresh(db_activity, attribute_names=['customer'])
    return db_activity

def get_user_activity(db: Session, activity_id: int):
    """Получает активность по ID."""
    return db.query(models.UserActivity).options(joinedload(models.UserActivity.customer)).filter(models.UserActivity.id == activity_id).first()

def get_user_activities(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    customer_id: Optional[int] = None,
    action_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> Tuple[int, List[models.UserActivity]]:
    """
    Получает список активностей с пагинацией и фильтрацией.
    Фильтрует по customer_id, action_type и временному диапазону (timestamp).
    Возвращает кортеж: (общее количество записей по фильтрам, список активностей).
    """
    query = db.query(models.UserActivity)

    # Применяем фильтры
    if customer_id is not None:
        query = query.filter(models.UserActivity.customer_id == customer_id)
    if action_type:
        query = query.filter(models.UserActivity.action_type == action_type)
    if start_time:
        query = query.filter(models.UserActivity.timestamp >= start_time)
    if end_time:
        query = query.filter(models.UserActivity.timestamp <= end_time)

    # Считаем общее количество записей *после* применения фильтров
    total_count = query.count()

    # Применяем сортировку, пагинацию и загрузку связей к основному запросу
    activities = (
        query
        .options(joinedload(models.UserActivity.customer)) # Eager load customer
        .order_by(models.UserActivity.timestamp.desc()) # Сортируем по времени (сначала новые)
        .offset(skip)
        .limit(limit)
        .all()
    )

    return total_count, activities

# === UserActivity CRUD (заготовка) ===
# ... 

# === Anomaly CRUD ===

def create_anomaly(db: Session, anomaly: schemas.AnomalyCreate) -> models.Anomaly:
    """
    Creates a new anomaly record and adds it to the session.
    Includes the new 'severity' field.
    DOES NOT COMMIT. Commit should be handled by the caller.
    """
    # anomaly.model_dump() будет включать severity из схемы
    db_anomaly = models.Anomaly(**anomaly.model_dump())
    db.add(db_anomaly)
    db.flush() # Ensure the object gets an ID if needed before commit
    db.refresh(db_anomaly) # Refresh to get default values like timestamp
    return db_anomaly

def get_detected_anomalies(db: Session, skip: int = 0, limit: int = 100) -> List[models.Anomaly]:
    """
    Retrieves a list of detected anomalies with pagination.
    Orders by detection timestamp descending (newest first).
    """
    return db.query(models.Anomaly)             .order_by(models.Anomaly.detection_timestamp.desc())             .offset(skip)             .limit(limit)             .all()

# --- НОВАЯ ФУНКЦИЯ для получения одной аномалии ---
def get_anomaly(db: Session, anomaly_id: int) -> Optional[models.Anomaly]:
    """
    Retrieves a single anomaly by its ID.
    """
    return db.query(models.Anomaly).filter(models.Anomaly.id == anomaly_id).first()

# --- НОВАЯ ФУНКЦИЯ ---
def delete_all_anomalies(db: Session) -> int:
    """
    Deletes all records from the anomalies table.
    Returns the number of rows deleted.
    """
    try:
        num_rows_deleted = db.query(models.Anomaly).delete()
        db.commit()
        print(f"Deleted {num_rows_deleted} existing anomalies.")
        return num_rows_deleted
    except Exception as e:
        print(f"Error deleting anomalies: {e}")
        db.rollback()
        return 0
# --------------------

# --- CRUD для Настроек (Новое) ---

def get_setting(db: Session, key: str) -> Optional[models.Setting]:
    """Получает одну настройку по ее ключу."""
    return db.query(models.Setting).filter(models.Setting.key == key).first()

def get_all_settings(db: Session) -> Dict[str, str]:
    """Получает все настройки из БД и возвращает их как словарь {key: value}."""
    settings = db.query(models.Setting).all()
    return {setting.key: setting.value for setting in settings}

def update_setting(db: Session, key: str, value: str) -> models.Setting:
    """Обновляет существующую настройку или создает новую, если ключ не найден."""
    db_setting = get_setting(db, key)
    if db_setting:
        db_setting.value = value
    else:
        db_setting = models.Setting(key=key, value=value)
        db.add(db_setting)
    db.commit()
    db.refresh(db_setting)
    return db_setting

# --- НОВЫЕ CRUD функции для Графиков ---

def get_activity_counts_by_interval(db: Session, interval: str = 'day') -> List[Dict]:
    """
    Подсчитывает количество активностей пользователей, сгруппированных по временному интервалу.
    Использует strftime для совместимости с SQLite.
    Args:
        db: Сессия SQLAlchemy.
        interval: Интервал группировки ('day' или 'hour').
    Returns:
        Список словарей вида [{'interval': str_timestamp, 'count': int}].
    """
    print(f"CRUD: Getting activity counts by {interval} using strftime...")
    if interval == 'day':
        # Группировка по дню с использованием strftime
        label_func = func.strftime('%Y-%m-%d', models.UserActivity.timestamp).label('interval')
    elif interval == 'hour':
        # Группировка по часу с использованием strftime
        label_func = func.strftime('%Y-%m-%d %H:00:00', models.UserActivity.timestamp).label('interval')
    else:
        raise ValueError("Unsupported interval. Use 'day' or 'hour'.")

    # Запрос
    query = (
        db.query(
            label_func,
            func.count(models.UserActivity.id).label('count')
        )
        .group_by(label_func)
        .order_by(label_func)
    )

    results = query.all()
    # Результат strftime - это строка, поэтому isoformat не нужен
    return [{'interval': str(row.interval), 'count': row.count} for row in results]

def get_anomaly_counts_by_detector(db: Session) -> List[Dict]:
    """
    Подсчитывает количество обнаруженных аномалий, сгруппированных по имени детектора.
    Returns:
        Список словарей вида [{'detector_name': str, 'count': int}].
    """
    print("CRUD: Getting anomaly counts by detector...")
    # Запрос
    query = (
        db.query(
            models.Anomaly.detector_name.label('detector_name'),
            func.count(models.Anomaly.id).label('count')
        )
        .filter(models.Anomaly.detector_name.isnot(None))
        .group_by(models.Anomaly.detector_name)
        .order_by(models.Anomaly.detector_name)
    )

    results = query.all()
    return [{'detector_name': row.detector_name, 'count': row.count} for row in results]

# --- НОВАЯ CRUD функция для получения скоров аномалий ---
def get_anomaly_scores_by_detector(
    db: Session,
    detector_name: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 10000 # Ограничение на всякий случай
) -> List[float]:
    """Возвращает список скоров для указанного детектора."""
    query = db.query(models.Anomaly.anomaly_score)\
        .filter(models.Anomaly.detector_name == detector_name)\
        .filter(models.Anomaly.anomaly_score != None)

    if start_time:
        query = query.filter(models.Anomaly.detection_timestamp >= start_time)
    if end_time:
        query = query.filter(models.Anomaly.detection_timestamp <= end_time)

    results = query.limit(limit).all()
    return [score for (score,) in results]

# === НОВАЯ ФУНКЦИЯ ДЛЯ SCATTER PLOT ===
def get_feature_scatter_data(
    db: Session,
    entity_type: str = 'order', # Пока только 'order'
    feature_x: str = 'total_amount',
    feature_y: str = 'item_count',
    limit: int = 500
) -> List[Dict]:
    """
    Возвращает данные для диаграммы рассеяния признаков.
    Поддерживает entity_type='order' и заданные признаки.
    Загружает заказы и проверяет, являются ли они аномальными.
    """
    if entity_type != 'order':
        raise NotImplementedError(f"Scatter plot data is currently only supported for entity_type='order'. Got: {entity_type}")

    allowed_features = {
        'total_amount', 
        'item_count', 
        'total_quantity', 
        'hour_of_day', 
        'day_of_week'
    }
    if feature_x not in allowed_features or feature_y not in allowed_features:
        raise ValueError(f"Invalid feature name(s). Allowed features: {allowed_features}. Got: x='{feature_x}', y='{feature_y}'")

    # Определяем, нужно ли загружать items
    needs_items = 'item_count' in [feature_x, feature_y] or 'total_quantity' in [feature_x, feature_y]
    
    # Запрашиваем заказы
    orders_query = db.query(models.Order)
    if needs_items:
        orders_query = orders_query.options(subqueryload(models.Order.items))
    
    orders_query = orders_query.order_by(models.Order.id.desc()).limit(limit)
    orders = orders_query.all()

    # Получаем ID аномальных заказов для быстрой проверки
    anomalous_order_ids = { 
        anomaly.entity_id 
        for anomaly in db.query(models.Anomaly.entity_id)\
            .filter(models.Anomaly.entity_type == 'order')\
            .distinct()
    }

    # Функция для извлечения значения признака
    def get_feature_value(order: models.Order, feature_name: str):
        if feature_name == 'total_amount':
            return order.total_amount
        elif feature_name == 'item_count':
            return len(order.items) if order.items else 0
        elif feature_name == 'total_quantity':
            return sum(item.quantity for item in order.items) if order.items else 0
        elif feature_name == 'hour_of_day':
            return order.created_at.hour if order.created_at else None
        elif feature_name == 'day_of_week':
             # weekday() возвращает 0 для понедельника, 6 для воскресенья
            return order.created_at.weekday() if order.created_at else None
        return None # На случай, если что-то пойдет не так

    scatter_data = []
    for order in orders:
        val_x = get_feature_value(order, feature_x)
        val_y = get_feature_value(order, feature_y)
        is_anomaly = order.id in anomalous_order_ids

        # Пропускаем точки с None значениями (например, если created_at был Null)
        if val_x is not None and val_y is not None:
            scatter_data.append({
                "id": order.id,
                "x": float(val_x), # Приводим к float для консистентности
                "y": float(val_y),
                "is_anomaly": is_anomaly
            })

    return scatter_data
# =======================================

# ... (rest of the file) ... 